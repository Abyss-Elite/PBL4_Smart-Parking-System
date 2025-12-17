"use client";

import ShiftCell from "./ShiftCell";
import { addDays, format } from "date-fns";

const shifts = ["08:00-17:30", "17:30-00:00", "00:00-08:00"];

export default function WeeklySchedule({ weekStart, schedule }) {
  const days = [...Array(7)].map((_, i) => addDays(weekStart, i));

  return (
    <div className="mt-4 overflow-x-auto rounded-xl bg-white p-4 shadow">
      <table className="w-full min-w-max table-fixed border-collapse">
        <thead>
          <tr className="bg-gray-100">
            <th className="sticky left-0 z-10 border border-gray-200 bg-gray-50 p-3 text-left font-semibold text-gray-700">
              Ca / Thứ
            </th>
            {days.map((d) => (
              <th
                key={d.toString()}
                className="border border-gray-200 bg-gray-50 p-3 text-center font-semibold text-gray-700"
              >
                <div className="text-xs text-gray-500 uppercase">{format(d, "EEE")}</div>
                <div className="text-sm font-bold">{format(d, "dd/MM")}</div>
              </th>
            ))}
          </tr>
        </thead>

        <tbody>
          {shifts.map((shift) => (
            <tr key={shift}>
              <td className="sticky left-0 z-10 border border-gray-200 bg-gray-50 p-3 font-medium text-gray-700">
                {shift}
              </td>

              {days.map((day) => {
                // Lấy tất cả nhân viên cùng ngày + ca
                const employees = schedule
                  .filter((s) => s.date === format(day, "yyyy-MM-dd") && s.shift === shift)
                  .map((s) => s.employee);

                return <ShiftCell key={day.toString()} employees={employees} />;
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
