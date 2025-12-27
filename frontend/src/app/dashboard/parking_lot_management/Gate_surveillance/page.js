"use client";

import VehicleRecognitionResult from "@/components/parking/generalParking/VehicleRecognitionResult";
import { useState } from "react";
import { Tabs, TabsList, TabsTrigger, TabsContent } from "@/components/ui/tabs";
import { useSSE } from "@/utils/useSSE";
import { carAPI } from "@/api/car/carAPI";
import { da } from "date-fns/locale";

export default function GateSurveillance() {
  const [currentMonitoredCarIn, setCurrentMonitoredCarIn] = useState(null);
  const [currentMonitoredCarOut, setCurrentMonitoredCarOut] = useState(null);
  const [showBookingWarning, setShowBookingWarning] = useState(false);
  const [pendingCheckInData, setPendingCheckInData] = useState(null);
  const [errorMessage, setErrorMessage] = useState("");

  useSSE(
    "http://192.168.100.252:8083/api/parkingLot/currentPlate/checkIn/stream",
    "check-in-plate",
    (data) => {
      if (data?.status === "BOOKING_NOT_STARTED") {
        setPendingCheckInData(data); // lưu lại xe
        setShowBookingWarning(true);
        return;
      }

      setCurrentMonitoredCarIn({
        licensePlate: data.licensePlate,
        time: data.car.checkInTime,
        image: data.car.imageInUrl,
      });
    }
  );

  useSSE(
    "http://192.168.100.252:8083/api/parkingLot/currentPlate/checkOut/stream",
    "check-out-plate",
    (data) => {
      setCurrentMonitoredCarOut({
        licensePlate: data.licensePlate,
        time: data.car.checkOutTime,
        image: data.car.imageOutUrl,
        fee: data.fee,
        id: data.car.id,
        isPaid: data.car.isPaid,
      });
    }
  );

  const handleConfirmFreeLot = async () => {
    try {
      await carAPI.checkInFreeLot();

      setShowBookingWarning(false);
      setPendingCheckInData(null);
    } catch (err) {
      console.error("Check-in free lot failed", err);
      setErrorMessage("Không thể chuyển sang bãi đỗ xe tự do. Vui lòng thử lại.");
      setTimeout(() => setErrorMessage(""), 4000);
    }
  };

  const handleCancelFreeLot = () => {
    setShowBookingWarning(false);
    setPendingCheckInData(null);
  };

  const handleRetake = () => alert("Chụp lại ảnh xe!");

  return (
    <>
      {errorMessage && (
        <div className="fixed top-4 right-4 z-50 flex items-center gap-2 rounded-lg bg-red-600 px-4 py-3 text-white shadow-lg">
          <span></span>
          <span>{errorMessage}</span>
        </div>
      )}

      {showBookingWarning && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/40">
          <div className="w-[420px] rounded-lg bg-white p-6 shadow-xl">
            <h3 className="mb-3 text-lg font-bold text-red-600"> Chưa tới giờ check-in</h3>

            <p className="mb-6 text-gray-700">
              Xe đã đặt booking nhưng chưa tới thời gian check-in.
              <br />
              Bạn có muốn chuyển sang <b>bãi đỗ xe tự do</b> không?
            </p>

            <div className="flex justify-end gap-3">
              <button
                onClick={handleCancelFreeLot}
                className="cursor-pointer rounded-lg border px-4 py-2 hover:bg-gray-100"
              >
                Không
              </button>

              <button
                onClick={handleConfirmFreeLot}
                className="cursor-pointer rounded-lg bg-green-600 px-4 py-2 text-white hover:bg-green-700"
              >
                Có
              </button>
            </div>
          </div>
        </div>
      )}

      <Tabs defaultValue="in" className="w-full">
        <TabsList className="grid w-full grid-cols-2">
          <TabsTrigger value="in" className="cursor-pointer">
            Camera vào
          </TabsTrigger>
          <TabsTrigger value="out" className="cursor-pointer">
            Camera ra
          </TabsTrigger>
        </TabsList>

        <TabsContent value="in" className="cursor-pointer">
          <VehicleRecognitionResult
            image={currentMonitoredCarIn?.image}
            time={currentMonitoredCarIn?.time}
            isOut={false}
            data={currentMonitoredCarIn}
            onRetake={handleRetake}
          />
        </TabsContent>

        <TabsContent value="out" className="cursor-pointer">
          <VehicleRecognitionResult
            image={currentMonitoredCarOut?.image}
            time={currentMonitoredCarOut?.time}
            isOut={true}
            data={currentMonitoredCarOut}
            onRetake={handleRetake}
            id={currentMonitoredCarOut?.id}
            isPaid={currentMonitoredCarOut?.isPaid}
          />
        </TabsContent>
      </Tabs>
    </>
  );
}
