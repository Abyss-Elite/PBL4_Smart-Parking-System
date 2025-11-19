"use client";
import VehicleRecognitionResult from "@/components/parking/generalParking/VehicleRecognitionResult";
import { carAPI } from "@/api/car/carAPI";
import { useEffect, useState } from "react";

export default function GateSurveillance() {
  const [currentMonitoredCar, setCurrentMonitoredCar] = useState({});

  useEffect(() => {
    const fetchData = async () => {
      const res = await carAPI.getCurrentMonitoredCar();
      setCurrentMonitoredCar(res.data);
    };
    fetchData();
  }, []);

  const linkVideo = "http://192.168.43.111:5000/video_feed";

  const handleRetake = () => alert("Chụp lại ảnh xe!");

  return (
    <div className="p-6">
      <VehicleRecognitionResult data={currentMonitoredCar} onRetake={handleRetake} linkVideo={linkVideo} />
    </div>
  );
}
