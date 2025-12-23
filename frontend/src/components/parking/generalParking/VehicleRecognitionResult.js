"use client";

import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import { ZoomIn, Camera, LogIn, LogOut, Clock, Car, ClipboardCheck } from "lucide-react";
import { formatDateTime } from "@/utils/formatDateTime";
import { useRouter } from "next/navigation";
import PATH from "@/routes/PATH";

export default function VehicleRecognitionResult({
  image,
  time,
  data,
  isOut,
  onRetake,
  linkVideo,
}) {
  const [isZoomed, setIsZoomed] = useState(false);
  const router = useRouter();

  const handlePayment = () => {
    router.push(`${PATH.PARKING_RESERVATION.PAYMENT}?bookingId=${data?.id}&amount=${data?.fee}`);
  };

  return (
    <div className="grid grid-cols-1 gap-6 md:grid-cols-2">
      <Card className="flex flex-col items-center justify-center shadow-xl">
        <CardHeader>
          <CardTitle className="text-center text-lg font-bold">Ảnh xe</CardTitle>
        </CardHeader>
        <CardContent className="flex flex-col items-center">
          <div className="relative overflow-hidden rounded-lg border shadow-md">
            <img
              src={image || "/no-camera.png"}
              alt="No Camera"
              width={420}
              height={280}
              className="rounded-lg border border-gray-200 object-cover shadow-sm"
            />
          </div>

          <div className="mt-4 flex space-x-3">
            <Button
              variant="outline"
              onClick={() => onRetake && onRetake()}
              className="flex items-center gap-2"
            >
              <Camera className="h-4 w-4" />
              Chụp lại
            </Button>

            <Button
              variant="secondary"
              onClick={() => setIsZoomed(!isZoomed)}
              className="flex items-center gap-2"
            >
              <ZoomIn className="h-4 w-4" />
              {isZoomed ? "Thu nhỏ" : "Phóng to"}
            </Button>
          </div>
        </CardContent>
      </Card>

      <Card className="shadow-xl">
        <CardHeader>
          <CardTitle className="text-lg font-bold">📋 Thông tin nhận diện</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4 text-base">
          <div className="flex items-center gap-2">
            <Clock className="h-5 w-5 text-gray-500" />
            <span className="font-semibold">Thời gian:</span>
            <span>{time ? formatDateTime(time) : "—"}</span>
          </div>

          <div className="flex items-center gap-2">
            <Car className="h-5 w-5 text-blue-600" />
            <span className="font-semibold">Biển số:</span>
            <span className="text-lg font-bold text-blue-700">
              {data?.licensePlate || "Không xác định"}
            </span>
          </div>

          <div className="flex items-center gap-2">
            {isOut === false ? (
              <LogIn className="h-5 w-5 text-green-600" />
            ) : (
              <LogOut className="h-5 w-5 text-red-600" />
            )}
            <span className="font-semibold">Hướng di chuyển:</span>
            <span
              className={`rounded px-2 py-1 font-semibold ${
                isOut === false ? "bg-green-100 text-green-700" : "bg-red-100 text-red-700"
              }`}
            >
              {isOut === false ? "Vào bãi" : "Ra khỏi bãi"}
            </span>
            {isOut === true && (
              <>
                <div>
                  <span>Tổng tiền:</span>
                  <span className="text-green-600">{data?.fee?.toLocaleString("vi-VN")}₫</span>
                </div>
                <button
                  onClick={handlePayment}
                  className="mt-6 w-full cursor-pointer rounded-lg bg-green-600 px-6 py-3 text-white shadow-md transition hover:bg-green-700"
                >
                  Thanh toán
                </button>
              </>
            )}
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
