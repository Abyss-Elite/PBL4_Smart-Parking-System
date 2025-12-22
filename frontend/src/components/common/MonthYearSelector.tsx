"use client";

import { useState, useEffect } from "react";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { getDaysInMonth } from "date-fns";
import { Calendar } from "lucide-react";

type Props = {
  onChange: (payload: {
    month: number;
    year: number;
    daysInMonth: number;
  }) => void;
};

export default function MonthYearSelector({ onChange }: Props) {
  const now = new Date();
  const [month, setMonth] = useState<number>(now.getMonth() + 1);
  const [year, setYear] = useState<number>(now.getFullYear());

  useEffect(() => {
    const date = new Date(year, month - 1, 1);

    onChange({
      month,
      year,
      daysInMonth: getDaysInMonth(date),
    });
  }, [month, year]);

  const monthsOptions = Array.from({ length: 12 }, (_, i) => i + 1);

  const yearsOptions = Array.from(
    { length: 10 },
    (_, i) => now.getFullYear() - 5 + i
  );

  return (
    <div className="flex items-center gap-4 rounded-xl bg-white p-4 shadow-lg dark:bg-neutral-900">
      <div className="flex items-center gap-3 rounded-lg bg-gray-100 px-4 py-2 dark:bg-neutral-800">
        <Calendar size={18} className="text-gray-500 dark:text-gray-300" />

        <Select value={String(month)} onValueChange={(v) => setMonth(Number(v))}>
          <p>Tháng</p>
          <SelectTrigger className="w-24 bg-white dark:bg-neutral-700">
            <SelectValue placeholder="Tháng" />
          </SelectTrigger>
          <SelectContent className="">
            {monthsOptions.map((m) => (
              <SelectItem className="" key={m} value={String(m)}>
                {m}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>

        <Select value={String(year)} onValueChange={(v) => setYear(Number(v))}>
          <p>Năm</p>
          <SelectTrigger className="w-28 bg-white dark:bg-neutral-700">
            <SelectValue placeholder="Năm" />
          </SelectTrigger>
          <SelectContent className="">
            {yearsOptions.map((y) => (
              <SelectItem className="" key={y} value={String(y)}>
                {y}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      </div>
    </div>
  );

}
