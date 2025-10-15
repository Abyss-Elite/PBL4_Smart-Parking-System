/** @type {import('next').NextConfig} */
const nextConfig = {
  images: {
    domains: [
      "cdn.dribbble.com",
      "photo.znews.vn",
      "upload.wikimedia.org",
      "192.168.1.14", // (tuỳ chọn) ESP32-CAM
      "192.168.1.12", // (tuỳ chọn) AI server
    ],
  },
};

export default nextConfig;