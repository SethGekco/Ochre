// qshp - Westwood SHP (Tiberian Sun / Red Alert 2) support for Qt.
// Copyright (C) 2026 Ochre contributors.
// Licensed under the GNU General Public License v3 or later.
//
// qshp-convert: SHP <-> sprite-sheet image, for editors that can open an
// SHP through the plugin but cannot save one.
//
//   qshp-convert unit.shp unit.png --palette unittem.pal
//   qshp-convert unit.png unit.shp --palette unittem.pal --frames 24
//
// The sheet is the plugin's layout: frames left to right, top to bottom,
// each the SHP's full canvas, 8 columns when the count is a multiple of 8.

#include "shpcodec.h"

#include <QtCore/QFile>
#include <QtCore/QFileInfo>
#include <QtGui/QGuiApplication>
#include <QtGui/QImage>

#include <cstdio>
#include <cstring>

namespace {

int usage()
{
    std::fprintf(stderr,
                 "usage: qshp-convert IN OUT [--palette FILE] [--frames N] [--columns C]\n"
                 "  .shp to an image: every frame on one sprite sheet\n"
                 "  an image to .shp: cut the sheet into N frames (default 1)\n"
                 "  --palette  .pal (6-bit VGA or JASC); default $QSHP_PALETTE, else grey\n");
    return 2;
}

QImage toImage(const shp::Sprite& sp, const shp::Palette& pal, int cols)
{
    const int n = int(sp.frames.size());
    const int rows = (n + cols - 1) / cols;
    QImage out(sp.width * cols, sp.height * rows, QImage::Format_Indexed8);
    QVector<QRgb> table(256);
    for (int i = 0; i < 256; ++i)
        table[i] = pal.argb[i];
    out.setColorTable(table);
    out.fill(0);
    for (int i = 0; i < n; ++i) {
        const std::vector<uint8_t> cell = shp::canvas(sp, size_t(i));
        const int ox = (i % cols) * sp.width, oy = (i / cols) * sp.height;
        for (int y = 0; y < sp.height; ++y)
            std::memcpy(out.scanLine(oy + y) + ox, cell.data() + size_t(y) * sp.width, size_t(sp.width));
    }
    return out;
}

} // namespace

int main(int argc, char** argv)
{
    qputenv("QT_QPA_PLATFORM", qgetenv("QT_QPA_PLATFORM").isEmpty() ? "offscreen" : qgetenv("QT_QPA_PLATFORM"));
    QGuiApplication app(argc, argv);
    QStringList args = app.arguments().mid(1);
    QString palettePath = qEnvironmentVariable("QSHP_PALETTE");
    int frames = 1, columns = 0;
    QStringList files;
    for (int i = 0; i < args.size(); ++i) {
        const QString a = args[i];
        if ((a == QLatin1String("--palette") || a == QLatin1String("--frames") || a == QLatin1String("--columns"))
            && i + 1 < args.size()) {
            const QString v = args[++i];
            if (a == QLatin1String("--palette"))
                palettePath = v;
            else if (a == QLatin1String("--frames"))
                frames = v.toInt();
            else
                columns = v.toInt();
        } else if (a.startsWith(QLatin1String("--"))) {
            return usage();
        } else {
            files << a;
        }
    }
    if (files.size() != 2 || frames < 1 || columns < 0)
        return usage();

    shp::Palette pal = shp::fallbackPalette();
    if (!palettePath.isEmpty() && !shp::loadPalette(QFile::encodeName(palettePath).toStdString(), pal)) {
        std::fprintf(stderr, "cannot read palette %s\n", qPrintable(palettePath));
        return 1;
    }
    if (palettePath.isEmpty())
        std::fprintf(stderr, "note: no --palette; using a grey stand-in\n");
    pal = shp::uniqueColours(pal);      // the plugin's colours, so its sheets convert back exactly

    const QString in = files[0], out = files[1];
    if (QFileInfo(in).suffix().toLower() == QLatin1String("shp")) {
        QFile f(in);
        if (!f.open(QIODevice::ReadOnly)) {
            std::fprintf(stderr, "cannot read %s\n", qPrintable(in));
            return 1;
        }
        const QByteArray bytes = f.readAll();
        shp::Sprite sp;
        std::string error;
        if (!shp::parse(reinterpret_cast<const uint8_t*>(bytes.constData()), size_t(bytes.size()), sp, &error)) {
            std::fprintf(stderr, "%s: %s\n", qPrintable(in), error.c_str());
            return 1;
        }
        const int cols = columns > 0 ? columns : shp::autoColumns(sp.frames.size());
        if (!toImage(sp, pal, cols).save(out)) {
            std::fprintf(stderr, "cannot write %s\n", qPrintable(out));
            return 1;
        }
        std::printf("%s: %zu frames of %dx%d, %d per row\n", qPrintable(out), sp.frames.size(), sp.width,
                    sp.height, cols);
        return 0;
    }

    QImage image(in);
    if (image.isNull()) {
        std::fprintf(stderr, "cannot read image %s\n", qPrintable(in));
        return 1;
    }
    const int cols = qMin(frames, columns > 0 ? columns : shp::autoColumns(size_t(frames)));
    const int rows = (frames + cols - 1) / cols;
    if (image.width() % cols || image.height() % rows) {
        std::fprintf(stderr, "%dx%d does not divide into %d columns x %d rows\n", image.width(),
                     image.height(), cols, rows);
        return 1;
    }
    const int w = image.width() / cols, h = image.height() / rows;
    // Same rule as the plugin's writer: a sheet whose colour table is the
    // palette keeps its indices; anything else is matched by colour.
    std::vector<uint8_t> idx(size_t(image.width()) * image.height(), 0);
    bool same = image.format() == QImage::Format_Indexed8 && image.colorCount() == 256;
    for (int i = 1; i < 256 && same; ++i)
        same = (image.color(i) & 0xFFFFFFu) == (pal.argb[i] & 0xFFFFFFu);
    if (same) {
        for (int y = 0; y < image.height(); ++y)
            std::memcpy(idx.data() + size_t(y) * image.width(), image.constScanLine(y), size_t(image.width()));
    } else {
        const QImage argb = image.convertToFormat(QImage::Format_ARGB32);
        QHash<QRgb, uint8_t> memo;
        for (int y = 0; y < argb.height(); ++y) {
            const QRgb* row = reinterpret_cast<const QRgb*>(argb.constScanLine(y));
            for (int x = 0; x < argb.width(); ++x) {
                if (qAlpha(row[x]) < 128)
                    continue;
                auto it = memo.constFind(row[x]);
                if (it == memo.constEnd())
                    it = memo.insert(row[x], shp::nearest(pal, row[x]));
                idx[size_t(y) * argb.width() + x] = it.value();
            }
        }
    }
    std::vector<std::vector<uint8_t>> canvases(size_t(frames), std::vector<uint8_t>(size_t(w) * h));
    for (int i = 0; i < frames; ++i) {
        const int ox = (i % cols) * w, oy = (i / cols) * h;
        for (int y = 0; y < h; ++y)
            std::memcpy(canvases[size_t(i)].data() + size_t(y) * w, idx.data() + size_t(oy + y) * image.width() + ox,
                        size_t(w));
    }
    const std::vector<uint8_t> bytes = shp::build(w, h, canvases);
    QFile f(out);
    if (!f.open(QIODevice::WriteOnly) || f.write(reinterpret_cast<const char*>(bytes.data()), qint64(bytes.size()))
                                             != qint64(bytes.size())) {
        std::fprintf(stderr, "cannot write %s\n", qPrintable(out));
        return 1;
    }
    std::printf("%s: %d frames of %dx%d\n", qPrintable(out), frames, w, h);
    return 0;
}
