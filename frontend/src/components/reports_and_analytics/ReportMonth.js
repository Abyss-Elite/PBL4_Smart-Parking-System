"use client";
import { useState } from "react";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";

export function ReportMonth() {
  const initialData = [
    { month: "01/2025", total: 320, fee: 6400000, electricity: 800000 },
    { month: "02/2025", total: 280, fee: 5600000, electricity: 750000 },
    { month: "03/2025", total: 410, fee: 10250000, electricity: 900000 },
    { month: "11/2025", total: 410, fee: 10250000, electricity: 900000 },
  ];

  const [data, setData] = useState(initialData);
  const [selectedMonth, setSelectedMonth] = useState(initialData[0].month);

  const currentItem = data.find((d) => d.month === selectedMonth);

  const isCurrentMonth = (month) => {
    const [m, y] = month.split("/");
    const now = new Date();
    return now.getMonth() + 1 === Number(m) && now.getFullYear() === Number(y);
  };

  const handleElectricityChange = (value) => {
    const newData = data.map((item) =>
      item.month === selectedMonth ? { ...item, electricity: value } : item
    );
    setData(newData);
  };

  return (
    <Card className="rounded-2xl border shadow-sm">
      <CardHeader>
        <CardTitle>Báo cáo theo tháng</CardTitle>
      </CardHeader>
      <CardContent className="space-y-6">
        {/* Chọn tháng */}
        <div className="flex items-center gap-4">
          <label htmlFor="monthSelect" className="font-medium text-gray-700">
            Chọn tháng:
          </label>
          <select
            id="monthSelect"
            value={selectedMonth}
            onChange={(e) => setSelectedMonth(e.target.value)}
            className="rounded-lg border border-gray-300 bg-white p-2 shadow-sm focus:outline-none"
          >
            {data.map((item) => (
              <option key={item.month} value={item.month}>
                {item.month}
              </option>
            ))}
          </select>
        </div>

        {/* Hiển thị thông tin tháng */}
        {currentItem && (
          <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
            <Card className="rounded-xl border bg-gray-50 p-4 shadow-sm">
              <p>
                <span className="font-semibold">Tổng lượt xe:</span> {currentItem.total}
              </p>
              <p>
                <span className="font-semibold">Tiền phí thu vào:</span>{" "}
                {currentItem.fee.toLocaleString()} đ
              </p>
              <p className="flex items-center gap-2">
                <span className="font-semibold">Tiền điện:</span>
                {isCurrentMonth(currentItem.month) ? (
                  <input
                    type="number"
                    value={currentItem.electricity}
                    onChange={(e) => handleElectricityChange(Number(e.target.value))}
                    className="w-32 rounded-md border p-1 shadow-sm focus:border-indigo-500 focus:ring-2 focus:ring-indigo-500 focus:outline-none"
                  />
                ) : (
                  <span>{currentItem.electricity.toLocaleString()} đ</span>
                )}
              </p>
              <p>
                <span className="font-semibold">Doanh thu thực:</span>{" "}
                {(currentItem.fee - currentItem.electricity).toLocaleString()} đ
              </p>
            </Card>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
