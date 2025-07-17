const sharp = require('sharp');
const fs = require('fs');
const path = require('path');

const IMG_DIR = './images';
const RESIZE_DIR = './images-optimized';

const WIDTHS = [540, 1320];
const WIDTH_2X = WIDTHS.map(width => width * 2);
const ALL_WIDTHS = [...WIDTHS, ...WIDTH_2X];

const JPEG_OPTIONS = {
    quality: 85,
    progressive: true,
    chromaSubsampling: '4:4:4',
    mozjpeg: true,
    force: true,
};

const WEBP_OPTIONS = {
    quality: 80,
    force: true,
};

function resizeImage(filename, width) {
    const imgPath = path.join(IMG_DIR, filename);
    const resizedFilename = `${path.basename(filename, path.extname(filename))}-${width}.webp`;
    const resizedPath = path.join(RESIZE_DIR, resizedFilename);
    sharp(imgPath).resize({ width }).webp(WEBP_OPTIONS).toFile(resizedPath);
}

function compressImage(filename) {
    const imgPath = path.join(IMG_DIR, filename);
    const compressedFilename = `${path.basename(filename, path.extname(filename))}-full.webp`;
    const compressedPath = path.join(RESIZE_DIR, compressedFilename);
    sharp(imgPath).webp(WEBP_OPTIONS).toFile(compressedPath);
}

if (!fs.existsSync(RESIZE_DIR)) fs.mkdirSync(RESIZE_DIR);

fs.readdirSync(IMG_DIR).forEach(file => {
    ALL_WIDTHS.forEach(width => resizeImage(file, width));
    compressImage(file);
});
