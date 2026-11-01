// qshp - Westwood SHP (Tiberian Sun / Red Alert 2) support for Qt.
// Copyright (C) 2026 Ochre contributors.
// Licensed under the GNU General Public License v3 or later.
//
// A Qt image-format plugin: drop the library into an application's
// imageformats/ folder and every QImageReader in it can open .shp sprites,
// whether or not the application knows the format exists.
//
// An SHP names no palette, so one is configured (see settings() below), and
// an SHP holds many frames while most editors open one image, so by default
// the frames are laid out as a sprite sheet. Images read here are
// Format_Indexed8: the palette indices survive for any program that keeps
// them, and the colour table makes them look right for the rest.

#include "shpcodec.h"

#include <QtCore/QCoreApplication>
#include <QtCore/QDir>
#include <QtCore/QFileInfo>
#include <QtCore/QSettings>
#include <QtCore/QStandardPaths>
#include <QtGui/QImage>
#include <QtGui/QImageIOHandler>
#include <QtGui/QImageIOPlugin>

#include <cstring>

#if defined(Q_OS_WIN)
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace {

// The folder this plugin was loaded from, so a palette and qshp.ini can sit
// right beside it.
QString pluginDir()
{
#if defined(Q_OS_WIN)
    HMODULE module = nullptr;
    wchar_t path[MAX_PATH];
    if (GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                           reinterpret_cast<LPCWSTR>(&pluginDir), &module)
        && GetModuleFileNameW(module, path, MAX_PATH))
        return QFileInfo(QString::fromWCharArray(path)).absolutePath();
    return QString();
#else
    Dl_info info;
    if (dladdr(reinterpret_cast<void*>(&pluginDir), &info) && info.dli_fname)
        return QFileInfo(QString::fromLocal8Bit(info.dli_fname)).absolutePath();
    return QString();
#endif
}

enum class Layout { Sheet, First, Frames };

struct Settings {
    shp::Palette palette;
    Layout layout = Layout::Sheet;
    int columns = 0;                    // 0 = automatic
};

// Where the settings come from, first found wins for each key:
//   QSHP_PALETTE, QSHP_FRAMES, QSHP_COLUMNS   environment variables
//   qshp.ini beside the plugin, then ~/.config/qshp/qshp.ini
//       [SHP] Palette=unittem.pal  Frames=sheet|first|frames  Columns=0
//   qshp.pal beside the plugin, else the only .pal file beside it
// A relative Palette path is taken from the ini file's own folder.
Settings settings()
{
    Settings s;
    const QString here = pluginDir();
    QStringList inis;
    if (!here.isEmpty())
        inis << QDir(here).filePath(QStringLiteral("qshp.ini"));
    inis << QDir(QStandardPaths::writableLocation(QStandardPaths::GenericConfigLocation))
                .filePath(QStringLiteral("qshp/qshp.ini"));

    QString palette = qEnvironmentVariable("QSHP_PALETTE");
    QString frames = qEnvironmentVariable("QSHP_FRAMES");
    QString columns = qEnvironmentVariable("QSHP_COLUMNS");
    for (const QString& ini : inis) {
        if (!QFileInfo::exists(ini))
            continue;
        QSettings file(ini, QSettings::IniFormat);
        const QString dir = QFileInfo(ini).absolutePath();
        const QString p = file.value(QStringLiteral("SHP/Palette")).toString();
        if (palette.isEmpty() && !p.isEmpty())
            palette = QDir(dir).absoluteFilePath(p);
        if (frames.isEmpty())
            frames = file.value(QStringLiteral("SHP/Frames")).toString();
        if (columns.isEmpty())
            columns = file.value(QStringLiteral("SHP/Columns")).toString();
    }
    if (palette.isEmpty() && !here.isEmpty()) {
        const QDir dir(here);
        if (dir.exists(QStringLiteral("qshp.pal")))
            palette = dir.filePath(QStringLiteral("qshp.pal"));
        else {
            const QStringList pals = dir.entryList({QStringLiteral("*.pal"), QStringLiteral("*.PAL")}, QDir::Files);
            if (pals.size() == 1)
                palette = dir.filePath(pals.first());
        }
    }
    if (palette.isEmpty() || !shp::loadPalette(QFile::encodeName(palette).toStdString(), s.palette))
        s.palette = shp::fallbackPalette();
    s.palette = shp::uniqueColours(s.palette);
    const QString f = frames.trimmed().toLower();
    s.layout = f == QLatin1String("first") ? Layout::First
             : f == QLatin1String("frames") ? Layout::Frames : Layout::Sheet;
    s.columns = qMax(0, columns.toInt());
    return s;
}

QVector<QRgb> colourTable(const shp::Palette& palette)
{
    QVector<QRgb> table(256);
    for (int i = 0; i < 256; ++i)
        table[i] = palette.argb[i];
    return table;
}

bool peekSniff(QIODevice* device)
{
    if (!device)
        return false;
    const QByteArray head = device->peek(int(shp::kHeader));
    if (head.size() < int(shp::kHeader))
        return false;
    const auto* h = reinterpret_cast<const uint8_t*>(head.constData());
    const QByteArray table = device->peek(int(shp::tableBytes(h)));
    const size_t size = device->isSequential() ? 0 : size_t(device->size() - device->pos());
    return shp::sniff(reinterpret_cast<const uint8_t*>(table.constData()), size_t(table.size()), size);
}

class ShpHandler : public QImageIOHandler {
public:
    bool canRead() const override
    {
        if (loaded_)
            return current_ < frameCount();
        if (peekSniff(device())) {
            setFormat("shp");
            return true;
        }
        return false;
    }

    bool read(QImage* image) override
    {
        if (!load())
            return false;
        const shp::Sprite& sp = sprite_;
        const int n = int(sp.frames.size());
        QImage out;
        if (settings_.layout == Layout::Sheet) {
            const int cols = settings_.columns > 0 ? settings_.columns : shp::autoColumns(size_t(n));
            const int rows = (n + cols - 1) / cols;
            out = QImage(sp.width * cols, sp.height * rows, QImage::Format_Indexed8);
            if (out.isNull())
                return false;
            out.setColorTable(colourTable(settings_.palette));
            out.fill(0);
            for (int i = 0; i < n; ++i) {
                const std::vector<uint8_t> cell = shp::canvas(sp, size_t(i));
                const int ox = (i % cols) * sp.width, oy = (i / cols) * sp.height;
                for (int y = 0; y < sp.height; ++y)
                    std::memcpy(out.scanLine(oy + y) + ox, cell.data() + size_t(y) * sp.width, size_t(sp.width));
            }
            out.setText(QStringLiteral("shp.frames"), QString::number(n));
            out.setText(QStringLiteral("shp.columns"), QString::number(cols));
            current_ = n;                   // one image: the sheet
        } else {
            if (current_ >= n)
                return false;
            out = QImage(sp.width, sp.height, QImage::Format_Indexed8);
            if (out.isNull())
                return false;
            out.setColorTable(colourTable(settings_.palette));
            const std::vector<uint8_t> cell = shp::canvas(sp, size_t(current_));
            for (int y = 0; y < sp.height; ++y)
                std::memcpy(out.scanLine(y), cell.data() + size_t(y) * sp.width, size_t(sp.width));
            out.setText(QStringLiteral("shp.frames"), QString::number(n));
            out.setText(QStringLiteral("shp.frame"), QString::number(current_));
            current_ = settings_.layout == Layout::First ? n : current_ + 1;
        }
        out.setText(QStringLiteral("shp.palette"), QString::fromStdString(settings_.palette.name));
        *image = out;
        return true;
    }

    // Writing. The image is either one frame, or a sheet whose text keys
    // "shp.frames" (and optionally "shp.columns") say how to cut it -- the
    // keys read() sets, and what qshp-convert passes for an exported PNG.
    bool write(const QImage& image) override
    {
        settings_ = settings();
        const int n = qMax(1, image.text(QStringLiteral("shp.frames")).toInt());
        int cols = image.text(QStringLiteral("shp.columns")).toInt();
        if (cols <= 0)
            cols = shp::autoColumns(size_t(n));
        cols = qMin(cols, n);
        const int rows = (n + cols - 1) / cols;
        if (image.width() % cols || image.height() % rows)
            return false;
        const int w = image.width() / cols, h = image.height() / rows;
        if (w < 1 || h < 1 || w > 4096 || h > 4096)
            return false;
        const std::vector<uint8_t> indices = toIndices(image, settings_.palette);
        std::vector<std::vector<uint8_t>> canvases(size_t(n), std::vector<uint8_t>(size_t(w) * h));
        for (int i = 0; i < n; ++i) {
            const int ox = (i % cols) * w, oy = (i / cols) * h;
            for (int y = 0; y < h; ++y)
                std::memcpy(canvases[size_t(i)].data() + size_t(y) * w,
                            indices.data() + size_t(oy + y) * image.width() + ox, size_t(w));
        }
        const std::vector<uint8_t> bytes = shp::build(w, h, canvases);
        return device()->write(reinterpret_cast<const char*>(bytes.data()), qint64(bytes.size()))
            == qint64(bytes.size());
    }

    // Palette indices for an image: an indexed image whose colour table is
    // the palette keeps its indices exactly (index 0 included); anything else
    // is matched colour by colour, alpha under 128 becoming index 0.
    static std::vector<uint8_t> toIndices(const QImage& image, const shp::Palette& palette)
    {
        std::vector<uint8_t> out(size_t(image.width()) * image.height(), 0);
        if (image.format() == QImage::Format_Indexed8 && image.colorCount() == 256) {
            bool same = true;
            for (int i = 1; i < 256 && same; ++i)
                same = (image.color(i) & 0xFFFFFFu) == (palette.argb[i] & 0xFFFFFFu);
            if (same) {
                for (int y = 0; y < image.height(); ++y)
                    std::memcpy(out.data() + size_t(y) * image.width(), image.constScanLine(y), size_t(image.width()));
                return out;
            }
        }
        const QImage argb = image.convertToFormat(QImage::Format_ARGB32);
        QHash<QRgb, uint8_t> memo;
        for (int y = 0; y < argb.height(); ++y) {
            const QRgb* row = reinterpret_cast<const QRgb*>(argb.constScanLine(y));
            for (int x = 0; x < argb.width(); ++x) {
                const QRgb c = row[x];
                if (qAlpha(c) < 128)
                    continue;
                auto it = memo.constFind(c);
                if (it == memo.constEnd())
                    it = memo.insert(c, shp::nearest(palette, c));
                out[size_t(y) * argb.width() + x] = it.value();
            }
        }
        return out;
    }

    int imageCount() const override
    {
        if (!const_cast<ShpHandler*>(this)->load())
            return 0;
        return settings_.layout == Layout::Frames ? frameCount() : 1;
    }

    bool jumpToImage(int index) override
    {
        if (!load() || settings_.layout != Layout::Frames || index < 0 || index >= frameCount())
            return false;
        current_ = index;
        return true;
    }

    bool jumpToNextImage() override { return jumpToImage(current_ + 1); }
    int currentImageNumber() const override { return current_; }
    int loopCount() const override { return -1; }
    int nextImageDelay() const override { return 100; }

    bool supportsOption(ImageOption option) const override { return option == Size; }

    // The size before decoding, so applications can refuse an oversized file
    // first. Read from the header without consuming the device.
    QVariant option(ImageOption option) const override
    {
        if (option != Size || !peekSniff(device()))
            return QVariant();
        const QByteArray head = device()->peek(int(shp::kHeader));
        const auto* h = reinterpret_cast<const uint8_t*>(head.constData());
        const int w = h[2] | (h[3] << 8), ht = h[4] | (h[5] << 8), n = h[6] | (h[7] << 8);
        const Settings s = settings();
        if (s.layout != Layout::Sheet)
            return QSize(w, ht);
        const int cols = s.columns > 0 ? s.columns : shp::autoColumns(size_t(n));
        return QSize(w * cols, ht * ((n + cols - 1) / cols));
    }

private:
    int frameCount() const { return int(sprite_.frames.size()); }

    bool load()
    {
        if (loaded_)
            return true;
        if (!device() || !peekSniff(device()))
            return false;
        const QByteArray bytes = device()->readAll();
        if (!shp::parse(reinterpret_cast<const uint8_t*>(bytes.constData()), size_t(bytes.size()), sprite_))
            return false;
        settings_ = settings();
        loaded_ = true;
        current_ = 0;
        return true;
    }

    shp::Sprite sprite_;
    Settings settings_;
    bool loaded_ = false;
    int current_ = 0;
};

} // namespace

class ShpPlugin : public QImageIOPlugin {
    Q_OBJECT
    Q_PLUGIN_METADATA(IID QImageIOHandlerFactoryInterface_iid FILE "qshp.json")

public:
    Capabilities capabilities(QIODevice* device, const QByteArray& format) const override
    {
        if (format == "shp")
            return CanRead | CanWrite;
        // No format named: claim the data only when the whole frame table
        // checks out. Never claim writing to an unnamed format.
        if (format.isEmpty() && device && device->isReadable() && peekSniff(device))
            return CanRead;
        return {};
    }

    QImageIOHandler* create(QIODevice* device, const QByteArray& format) const override
    {
        auto* handler = new ShpHandler;
        handler->setDevice(device);
        handler->setFormat(format.isEmpty() ? QByteArray("shp") : format);
        return handler;
    }
};

#include "qshpplugin.moc"
