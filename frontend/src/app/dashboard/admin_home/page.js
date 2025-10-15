"use client";

import { Button } from "@/components/ui/button";
import { useState } from "react";
import ParkingUsageChart from "@/components/adminPage/ui/ParkingUsageChart";
import RevenueChart from "@/components/adminPage/ui/RevenueChart";
import RecentVehicleTable from "@/components/adminPage/ui/RecentVehicleTable";
import ExpiringTicketList from "@/components/adminPage/ui/ExpiringTicketList";
import ItemCard from "@/components/adminPage/ui/ItemCard";
import PATH from "@/routes/PATH";
import { useRouter } from "next/navigation";

export default function AdminDashboard() {
  const router = useRouter();
  const [working, setWorking] = useState(false);

  const handleWork = () => {
    setWorking(true);
    setTimeout(() => setWorking(false), 2000);
    router.push(PATH.DASHBOARD.PARKING_LOT_MANAGEMENT.Gate_Surveillance);
  };
  const usagePercent = 85;

  const revenueData = [
    { month: "T1", revenue: 12000000 },
    { month: "T2", revenue: 18000000 },
    { month: "T3", revenue: 22000000 },
    { month: "T4", revenue: 16000000 },
    { month: "T5", revenue: 24000000 },
  ];

  const activityList = [
    { plate: "43A-12345", time: "08:42", status: "Vào" },
    { plate: "92B-67890", time: "09:10", status: "Ra" },
    { plate: "43C-22334", time: "09:45", status: "Vào" },
  ];

  const expiringTickets = [
    { apartment: "A101", plate: "43A-12345", expiryDate: "15/10/2025" },
    { apartment: "B204", plate: "92B-67890", expiryDate: "18/10/2025" },
  ];

  return (
    <div className="space-y-6 p-4">
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-semibold">Trang Admin</h1>
        <Button onClick={handleWork} className="bg-green-600 hover:bg-green-700">
          {working ? "Đang xử lý..." : "Giám sát"}
        </Button>
      </div>

      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <ItemCard title="Tổng xe đang gửi" content="126" classNameContent="text-2xl font-bold" />
        <ItemCard title="Căn hộ đăng ký" content="82" classNameContent="text-2xl font-bold" />
        <ItemCard
          title="Doanh thu tháng này"
          content="58,000,000₫"
          classNameContent="text-2xl font-bold text-green-600"
        />
        <ItemCard title="Chỗ đỗ còn trống" content="14" classNameContent="text-2xl font-bold" />
      </div>

      <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
        <RevenueChart revenueData={revenueData} />

        <ParkingUsageChart target={usagePercent} label="Đang sử dụng" />
      </div>

      <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
        <RecentVehicleTable activityList={activityList} />

        <ExpiringTicketList expiringTickets={expiringTickets} />
      </div>
    </div>
  );
}
