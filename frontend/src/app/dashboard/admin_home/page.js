"use client";

import { Button } from "@/components/ui/button";
import { useEffect, useState } from "react";
import ParkingUsageChart from "@/components/adminPage/ui/ParkingUsageChart";
import RevenueChart from "@/components/adminPage/ui/RevenueChart";
import RecentVehicleTable from "@/components/adminPage/ui/RecentVehicleTable";
import NextReservedCars from "@/components/adminPage/ui/NextReservedCars";
import ItemCard from "@/components/adminPage/ui/ItemCard";
import { formatCurrency } from "@/utils/formatCurrency";
import PATH from "@/routes/PATH";
import { useRouter } from "next/navigation";
import { carAPI } from "@/api/car/carAPI";
import { percentageToInt } from "@/utils/percentageToInt";
import { parkingLotAPI } from "@/api/parking-lot/parkingLotAPI";
import { revenueAPI } from "@/api/revenue/revenueAPI";

export default function AdminDashboard() {
  const router = useRouter();
  const [working, setWorking] = useState(false);
  const [totalBookedCars, setTotalBookedCars] = useState(0);
  const [parkingUsageInfo, setParkingUsageInfo] = useState({});
  const [revenueCurrentMonth, setRevenueCurrentMonth] = useState(0);
  const [recentActivitiesCars, setRecentActivitiesCars] = useState([]);
  const [nextReservedCars, setNextReservedCars] = useState([]);
  const [currentVehicleCondition, setCurrentVehicleCondition] = useState({});
  const [revenueData, setRevenueData] = useState();

  const yearNow = new Date().getFullYear();

  useEffect(() => {
    const fetchData = async () => {
      const res = await carAPI.getTotalBookedCars();
      const res2 = await parkingLotAPI.getUsageInfo();
      const res3 = await revenueAPI.revenueCurrentMonth();
      const res4 = await carAPI.getRecentActivitiesCar();
      const res5 = await carAPI.getNextReservedCars();
      const res6 = await parkingLotAPI.getCurrentVehicleCondition();
      const res7 = await revenueAPI.revenueYear(yearNow);
      setTotalBookedCars(res.data.totalBookedCars);
      setCurrentVehicleCondition(res6.data);
      setParkingUsageInfo(res2.data);
      setRevenueCurrentMonth(res3.data.revenue);
      setRecentActivitiesCars(res4.data);
      setNextReservedCars(res5.data);
      setRevenueData(res7.data);
    };
    fetchData();
  }, []);

  const handleWork = () => {
    setWorking(true);
    setTimeout(() => setWorking(false), 2000);
    router.push(PATH.DASHBOARD.PARKING_LOT_MANAGEMENT.Gate_Surveillance);
  };

  return (
    <div className="space-y-6 p-4">
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-semibold">Trang Admin</h1>
        <Button onClick={handleWork} className="bg-green-600 hover:bg-green-700 cursor-pointer">
          {working ? "Đang xử lý..." : "Giám sát"}
        </Button>
      </div>

      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <ItemCard
          title="Tổng xe đang gửi"
          content={currentVehicleCondition.totalCurrentCars}
          classNameContent="text-2xl font-bold"
        />
        <ItemCard
          title="Xe đăng ký chỗ trước"
          content={totalBookedCars}
          classNameContent="text-2xl font-bold"
        />
        <ItemCard
          title="Doanh thu tháng này"
          content={formatCurrency(revenueCurrentMonth)}
          classNameContent="text-2xl font-bold text-green-600"
        />
        <ItemCard
          title="Chỗ đỗ còn trống"
          content={currentVehicleCondition.totalAvailable}
          classNameContent="text-2xl font-bold"
        />
      </div>

      <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
        <RevenueChart revenueData={revenueData} />

        <ParkingUsageChart
          target={percentageToInt(parkingUsageInfo.usageRate)}
          label="Đang sử dụng"
        />
      </div>

      <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
        <RecentVehicleTable activityList={recentActivitiesCars} />

        <NextReservedCars nextReservedCars={nextReservedCars} />
      </div>
    </div>
  );
}
