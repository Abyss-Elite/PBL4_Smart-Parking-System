"use client";

import { useState, useEffect } from "react";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { addWeeks, startOfWeek, endOfWeek, format, isBefore, isAfter } from "date-fns";
import { ChevronLeft, ChevronRight, Calendar } from "lucide-react";

export default function WeekSelector({ onWeekChange, initialStartDate }) {
  const apiDate = new Date(initialStartDate);

  // Tuần đầu tiên cho phép
  const firstAvailableWeekStart = startOfWeek(apiDate, { weekStartsOn: 1 });

  // Tuần hiện tại thực tế
  const realWeekStart = startOfWeek(new Date(), { weekStartsOn: 1 });

  // FIX QUAN TRỌNG: Chọn tuần khởi tạo = tuần hiện tại (nếu hợp lệ)
  const initialWeek = isBefore(realWeekStart, firstAvailableWeekStart)
    ? firstAvailableWeekStart
    : realWeekStart;

  // State tuần hiện tại
  const [currentWeekStart, setCurrentWeekStart] = useState(initialWeek);

  // State tháng (theo tuần hiện tại)
  const [selectedMonth, setSelectedMonth] = useState(format(initialWeek, "MM/yyyy"));

  useEffect(() => {
    // Emit ra ngoài
    onWeekChange(currentWeekStart, endOfWeek(currentWeekStart, { weekStartsOn: 1 }));

    // Đồng bộ dropdown tháng
    setSelectedMonth(format(currentWeekStart, "MM/yyyy"));
  }, [currentWeekStart]);

  // =====================================================
  // Chuyển tuần
  // =====================================================
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

  // =====================================================
  // Chọn tháng → nhảy tới tuần đầu tiên của tháng đó
  // =====================================================
  const handleSelectMonth = (value) => {
    setSelectedMonth(value);

    const [month, year] = value.split("/").map(Number);
    const firstDay = new Date(year, month - 1, 1);

    let firstWeekOfMonth = startOfWeek(firstDay, { weekStartsOn: 1 });

    if (isBefore(firstWeekOfMonth, firstAvailableWeekStart)) {
      firstWeekOfMonth = firstAvailableWeekStart;
    }

    setCurrentWeekStart(firstWeekOfMonth);
  };

  // =====================================================
  // Tạo list tháng từ API → hiện tại (2 chữ số cho tháng)
  // =====================================================
  const monthsOptions = [];
  let tempDate = new Date(apiDate.getFullYear(), apiDate.getMonth(), 1);
  const now = new Date();

  while (tempDate <= now) {
    const monthStr = String(tempDate.getMonth() + 1).padStart(2, "0");
    const formattedMonth = `${monthStr}/${tempDate.getFullYear()}`;
    monthsOptions.push(formattedMonth);
    tempDate.setMonth(tempDate.getMonth() + 1);
  }

  const weekLabel = `${format(currentWeekStart, "dd MMM yyyy")} - ${format(
    endOfWeek(currentWeekStart, { weekStartsOn: 1 }),
    "dd MMM yyyy"
  )}`;

  return (
    <div className="flex flex-wrap items-center gap-4 rounded-xl bg-white p-4 shadow-lg dark:bg-neutral-900">
      <div className="flex items-center gap-3 rounded-lg bg-gray-100 px-4 py-2 dark:bg-neutral-800">
        <button
          className="rounded-full bg-white p-2 shadow hover:bg-gray-100 dark:bg-neutral-700 dark:hover:bg-neutral-600"
          onClick={handlePrevWeek}
        >
          <ChevronLeft size={18} />
        </button>

        <span className="text-sm font-semibold text-gray-800 sm:text-base dark:text-gray-100">
          {weekLabel}
        </span>

        <button
          className="rounded-full bg-white p-2 shadow hover:bg-gray-100 dark:bg-neutral-700 dark:hover:bg-neutral-600"
          onClick={handleNextWeek}
        >
          <ChevronRight size={18} />
        </button>
      </div>

      <div className="flex items-center gap-2">
        <Calendar className="text-gray-500 dark:text-gray-300" size={18} />

        <Select value={selectedMonth} onValueChange={handleSelectMonth}>
          <SelectTrigger className="w-40 rounded-lg border-gray-300 bg-white dark:border-neutral-700 dark:bg-neutral-800">
            <SelectValue placeholder="Chọn tháng" />
          </SelectTrigger>

          <SelectContent className="dark:bg-neutral-800 dark:text-white">
            {monthsOptions.map((m, idx) => (
              <SelectItem key={idx} value={m}>
                {m}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      </div>
    </div>
  );
}