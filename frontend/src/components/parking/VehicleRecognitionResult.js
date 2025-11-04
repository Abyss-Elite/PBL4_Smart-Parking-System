"use client";

import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import { ZoomIn, Camera, LogIn, LogOut, Clock, Car, ClipboardCheck } from "lucide-react";
import { formatDateTime } from "@/utils/formatDateTime";

export default function VehicleRecognitionResult({ data, onRetake, linkVideo }) {
  const [isZoomed, setIsZoomed] = useState(false);

  return (
    <div className="grid grid-cols-1 gap-6 md:grid-cols-2">
      <Card className="flex flex-col items-center justify-center shadow-xl">
        <CardHeader>
          <CardTitle className="text-center text-lg font-bold">Ảnh xe</CardTitle>
        </CardHeader>
        <CardContent className="flex flex-col items-center">
          <div className="relative overflow-hidden rounded-lg border shadow-md">
            {linkVideo ? (
              /* eslint-disable @next/next/no-img-element */
              <img
                src="http://192.168.43.86:5000/video_feed"
                alt="Live Feed"
                width={420}
                height={280}
                className={`rounded-lg border border-gray-200 object-cover shadow-sm transition-transform duration-300 ${
                  isZoomed ? "scale-110" : "scale-100"
                }`}
              />
            ) : (
              <img
                src="/placeholder-car.png"
                alt="No Camera"
                width={420}
                height={280}
                className="rounded-lg border border-gray-200 object-cover shadow-sm"
              />
              /* eslint-enable @next/next/no-img-element */
            )}
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
            <span>{formatDateTime(data?.dateTime) || "—"}</span>
          </div>

          <div className="flex items-center gap-2">
            <Car className="h-5 w-5 text-blue-600" />
            <span className="font-semibold">Biển số:</span>
            <span className="text-lg font-bold text-blue-700">
              {data?.currentPlate || "Không xác định"}
            </span>
          </div>

          <div className="flex items-center gap-2">
            <ClipboardCheck className="h-5 w-5 text-green-600" />
            <span className="font-semibold">Trạng thái:</span>
            <span
              className={`rounded px-2 py-1 font-semibold ${
                data?.status === "Đã xác nhận"
                  ? "bg-green-100 text-green-700"
                  : data?.status === "Đang chờ"
                    ? "bg-yellow-100 text-yellow-700"
                    : "bg-red-100 text-red-700"
              }`}
            >
              {data?.status || "—"}
            </span>
          </div>

          <div className="flex items-center gap-2">
            {data?.direction === "in" ? (
              <LogIn className="h-5 w-5 text-green-600" />
            ) : (
              <LogOut className="h-5 w-5 text-red-600" />
            )}
            <span className="font-semibold">Hướng di chuyển:</span>
            <span
              className={`rounded px-2 py-1 font-semibold ${
                data?.direction === "in" ? "bg-green-100 text-green-700" : "bg-red-100 text-red-700"
              }`}
            >
              {data?.direction === "in" ? "Vào bãi" : "Ra khỏi bãi"}
            </span>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
