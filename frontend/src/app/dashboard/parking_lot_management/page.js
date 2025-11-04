"use client";

import { useState, useEffect } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { MapPin, Camera, RefreshCcw } from "lucide-react";
import PATH from "@/routes/PATH";
import { useRouter } from "next/navigation";

export default function ParkingManagementPage() {
  const router = useRouter();
  const [currentTime, setCurrentTime] = useState("");
  const [usedSlots, setUsedSlots] = useState(37);
  const totalSlots = 50;
  const percentUsed = (usedSlots / totalSlots) * 100;
  const linkVideo = "http://192.168.43.86:5000/video_feed";

  useEffect(() => {
    const timer = setInterval(() => {
      const now = new Date();
      setCurrentTime(now.toLocaleString());
    }, 1000);
    return () => clearInterval(timer);
  }, []);

  const handleGate_surveillance = () => {
    router.push(PATH.DASHBOARD.PARKING_LOT_MANAGEMENT.Gate_Surveillance);
  };

  return (
    <div className="space-y-8 p-8">
      <h1 className="flex items-center gap-2 text-3xl font-bold text-blue-700">Quản lý Bãi Xe</h1>

      <div className="grid grid-cols-1 gap-6 md:grid-cols-3">
        <Card className="border-blue-200 shadow-sm cursor-pointer">
          <CardHeader>
            <CardTitle className="flex items-center gap-2 text-blue-700">
              <MapPin className="h-5 w-5" />
              Vị trí Bãi Xe
            </CardTitle>
          </CardHeader>
          <CardContent>
            <p className="mb-2 text-gray-700">
              Khu vực: <b>A1 - Tầng Hầm B</b>
            </p>
            <p className="mb-3 text-sm text-gray-500">Cập nhật: {currentTime}</p>
            <div className="flex h-32 w-full items-center justify-center rounded-xl border border-dashed border-blue-300 bg-blue-50 text-gray-500">
              Sơ đồ bãi xe (demo)
            </div>
          </CardContent>
        </Card>

        <Card className="border-green-200 shadow-sm cursor-pointer">
          <CardHeader>
            <CardTitle className="flex items-center gap-2 text-green-700">Sức chứa</CardTitle>
          </CardHeader>
          <CardContent>
            <p className="mb-2 text-gray-700">
              Tổng chỗ: <b>{totalSlots}</b>
            </p>
            <p className="mb-2 text-gray-700">
              Đang sử dụng: <b>{usedSlots}</b>
            </p>
            <Progress value={percentUsed} className="mt-3 h-3" />
            <p
              className={`mt-2 font-medium ${percentUsed > 80 ? "text-red-600" : "text-green-600"}`}
            >
              {percentUsed.toFixed(1)}% công suất
            </p>
            <Button
              variant="outline"
              size="sm"
              className="mt-4"
              onClick={() => {
                const randomUsed = Math.floor(Math.random() * totalSlots);
                setUsedSlots(randomUsed);
              }}
            >
              <RefreshCcw className="mr-2 h-4 w-4" /> Làm mới
            </Button>
          </CardContent>
        </Card>

        <Card className="border-yellow-200 shadow-sm cursor-pointer" onClick={handleGate_surveillance}>
          <CardHeader>
            <CardTitle className="flex items-center gap-2 text-yellow-700">
              <Camera className="h-5 w-5" />
              Trạng thái & Giám sát
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="mb-4 flex items-center gap-2">
              <Badge variant="outline" className="bg-green-100 text-green-700">
                Hoạt động
              </Badge>
              <span className="text-sm text-gray-500">Camera giám sát đang bật</span>
            </div>

            <div className="w-full overflow-hidden rounded-xl border border-gray-300 shadow-sm">
              <iframe src={linkVideo} className="h-64 w-full" allow="autoplay" />
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
