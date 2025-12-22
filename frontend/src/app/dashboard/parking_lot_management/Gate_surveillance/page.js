"use client";

import VehicleRecognitionResult from "@/components/parking/generalParking/VehicleRecognitionResult";
import { useEffect, useState } from "react";

export default function GateSurveillance() {
  const [currentMonitoredCar, setCurrentMonitoredCar] = useState(null);

  useEffect(() => {
    const es = new EventSource(
      "http://192.168.100.252:8083/api/parkingLot/currentPlate/stream"
    );

    const handler = (e) => {
      const data = JSON.parse(e.data);

      setCurrentMonitoredCar(data);

      console.log("SSE:", data);
    };

    es.addEventListener("current-plate", handler);

    es.onerror = (e) => {
      console.error("SSE error:", e);
      es.close();
    };

    return () => {
      es.removeEventListener("current-plate", handler);
      es.close();
    };
  }, []);

  const handleRetake = () => alert("Chụp lại ảnh xe!");

  return (
    <div className="p-6">
      <VehicleRecognitionResult
        data={currentMonitoredCar}
        onRetake={handleRetake}
      />
    </div>
  );
}
