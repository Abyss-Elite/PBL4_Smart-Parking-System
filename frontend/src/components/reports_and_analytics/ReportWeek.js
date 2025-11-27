"use client";
import { useState, useEffect } from "react";
import { addWeeks, startOfWeek, endOfWeek, format, isBefore, isAfter } from "date-fns";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import { revenueAPI } from "@/api/revenue/revenueAPI";

export function ReportWeek({ initialStartDate }) {
  const apiDate = new Date(initialStartDate);

  const firstAvailableWeekStart = startOfWeek(apiDate, { weekStartsOn: 1 });
  const realWeekStart = startOfWeek(new Date(), { weekStartsOn: 1 });

  const initialWeek = isBefore(realWeekStart, firstAvailableWeekStart)
    ? firstAvailableWeekStart
    : realWeekStart;

  const [currentWeekStart, setCurrentWeekStart] = useState(initialWeek);
  const [selectedMonth, setSelectedMonth] = useState(format(initialWeek, "MM/yyyy"));

  const [data, setData] = useState({
    carsOut: 0,
    carsIn: 0,
    carsInOut: 0,
    totalRevenue: 0,
    weekStart: "",
    weekEnd: "",
  });

  useEffect(() => {
    const fetchData = async () => {
      const dateStr = format(currentWeekStart, "yyyy-MM-dd");

      try {
        const res = await revenueAPI.revenueWeek(dateStr);
        setData(res.data);
      } catch (err) {
        console.error("Lỗi API tuần:", err);
      }
    };

    fetchData();
  }, [currentWeekStart]);

  useEffect(() => {
    setSelectedMonth(format(currentWeekStart, "MM/yyyy"));
  }, [currentWeekStart]);

  const handlePrevWeek = () => {
    const newStart = addWeeks(currentWeekStart, -1);
    if (!isBefore(newStart, firstAvailableWeekStart)) {
      setCurrentWeekStart(newStart);
    }
  };

  const handleNextWeek = () => {
    const newStart = addWeeks(currentWeekStart, 1);
    if (!isAfter(newStart, realWeekStart)) {
      setCurrentWeekStart(newStart);
    }
  };

  const handleSelectMonth = (value) => {
    setSelectedMonth(value);
    const [month, year] = value.split("/").map(Number);

    let firstWeekOfMonth = startOfWeek(new Date(year, month - 1, 1), { weekStartsOn: 1 });

    if (isBefore(firstWeekOfMonth, firstAvailableWeekStart)) {
      firstWeekOfMonth = firstAvailableWeekStart;
    }

    setCurrentWeekStart(firstWeekOfMonth);
  };

  const monthsOptions = [];
  let tempDate = new Date(apiDate.getFullYear(), apiDate.getMonth(), 1);
  const now = new Date();

  while (tempDate <= now) {
    const monthStr = String(tempDate.getMonth() + 1).padStart(2, "0");
    monthsOptions.push(`${monthStr}/${tempDate.getFullYear()}`);
    tempDate.setMonth(tempDate.getMonth() + 1);
  }

  const weekLabel = `${format(currentWeekStart, "dd MMM yyyy")} - ${format(
    endOfWeek(currentWeekStart, { weekStartsOn: 1 }),
    "dd MMM yyyy"
  )}`;

  return (
    <Card className="rounded-2xl border shadow-sm">
      <CardHeader>
        <CardTitle>Báo cáo theo tuần</CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="flex items-center gap-4">
          <label htmlFor="monthSelect" className="font-medium text-gray-700">
            Chọn tháng:
          </label>
          <select
            id="monthSelect"
            value={selectedMonth}
            onChange={(e) => handleSelectMonth(e.target.value)}
            className="rounded-lg border border-gray-300 bg-white p-2 shadow-sm focus:border-indigo-500 focus:ring-2 focus:ring-indigo-500 focus:outline-none"
          >
            {monthsOptions.map((m) => (
              <option key={m} value={m}>
                {m}
              </option>
            ))}
          </select>
        </div>

        <div className="rounded-xl border bg-gray-50 p-4 shadow-sm">
          <p>
            <span className="font-semibold">Tuần:</span> {weekLabel}
          </p>

          <p>
            <span className="font-semibold">Lượt xe vào:</span> {data.carsIn}
          </p>
          <p>
            <span className="font-semibold">Lượt xe ra:</span> {data.carsOut}
          </p>
          <p>
            <span className="font-semibold">Tổng lượt vào/ra:</span> {data.carsInOut}
          </p>
          <p>
            <span className="font-semibold">Tổng phí thu:</span>{" "}
            {data.totalRevenue?.toLocaleString()} đ
          </p>

          <div className="mt-2 flex gap-2">
            <button
              onClick={handlePrevWeek}
              className="rounded-md bg-gray-200 px-3 py-1 hover:bg-gray-300"
            >
              Tuần trước
            </button>
            <button
              onClick={handleNextWeek}
              className="rounded-md bg-gray-200 px-3 py-1 hover:bg-gray-300"
            >
              Tuần sau
            </button>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
