// qshp test helper: read files the way an application with no idea of the
// format does (QImageReader on a buffer, no format named) and print, per
// file, the format Qt picked and an MD5 of every frame's indices.
//   readcheck FILE...      (QSHP_FRAMES=frames for per-frame images)
#include <QtCore/QBuffer>
#include <QtCore/QCryptographicHash>
#include <QtCore/QFile>
#include <QtGui/QGuiApplication>
#include <QtGui/QImage>
#include <QtGui/QImageReader>
#include <QtGui/QImageWriter>
#include <cstdio>


int main(int argc, char** argv)
{
    QGuiApplication app(argc, argv);
    for (int a = 1; a < argc; ++a) {
        QFile f(QString::fromLocal8Bit(argv[a]));
        if (!f.open(QIODevice::ReadOnly)) { std::printf("UNREADABLE\t%s\n", argv[a]); continue; }
        QByteArray bytes = f.readAll();
        QBuffer buffer(&bytes);
        buffer.open(QIODevice::ReadOnly);
        // What Qt detects, asked the way it decides: by content alone.
        const QByteArray format = QImageReader::imageFormat(&buffer);
        QImageReader reader(&buffer);
        const QSize declared = reader.size();
        QCryptographicHash md5(QCryptographicHash::Md5);
        int frames = 0;
        QImage img;
        // read() moves on to the next image by itself; jumpToNextImage() would
        // skip one. Bounded by imageCount(): some readers keep saying canRead().
        const int count = qMax(1, reader.imageCount());
        while (frames < count && reader.read(&img)) {
            const int bytesPerLine = (img.width() * img.depth() + 7) / 8;
            for (int y = 0; y < img.height(); ++y)
                md5.addData(reinterpret_cast<const char*>(img.constScanLine(y)), bytesPerLine);
            ++frames;
            if (!reader.canRead())
                break;
        }
        std::printf("%s\t%d\t%s\t%dx%d\t%s\n", format.constData(), frames,
                    md5.result().toHex().constData(), declared.width(), declared.height(), argv[a]);
    }
    return 0;
}
