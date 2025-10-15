"use client";
import VehicleRecognitionResult from "@/components/parking/VehicleRecognitionResult";

export default function GateSurveillance() {
  const mockData = {
    imageUrl: "https://upload.wikimedia.org/wikipedia/commons/3/3b/Car.jpg",
    time: "2025-10-14 21:35:27",
    licensePlate: "43A-12345",
    status: "Đã xác nhận",
    direction: "in",
  };

  const handleRetake = () => alert("Chụp lại ảnh xe!");

  return (
    <div className="p-6">
      <VehicleRecognitionResult data={mockData} onRetake={handleRetake} />
    </div>
  );
}
