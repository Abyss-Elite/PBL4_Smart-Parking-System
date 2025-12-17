"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";

// Danh sách màu pastel nhạt (tailwind 300)
const colors = [
  "bg-blue-300",
  "bg-green-300",
  "bg-red-300",
  "bg-purple-300",
  "bg-yellow-300",
  "bg-pink-300",
  "bg-indigo-300",
  "bg-teal-300",
];

// Map lưu nhân viên -> màu
const employeeColorMap = new Map();

function getEmployeeColor(employee) {
  if (!employee) return "bg-gray-200";
  if (employeeColorMap.has(employee.id)) {
    return employeeColorMap.get(employee.id);
  }

  // Lấy màu chưa dùng
  const usedColors = Array.from(employeeColorMap.values());
  const availableColors = colors.filter((c) => !usedColors.includes(c));

  const color = availableColors.length > 0 ? availableColors[0] : colors[0]; // fallback nếu hết màu
  employeeColorMap.set(employee.id, color);
  return color;
}

export default function ShiftCell({ employees = [] }) {
  const [showTooltip, setShowTooltip] = useState(false);
  const router = useRouter();
  const hasEmployee = employees.length > 0;

  const handleClick = (employee) => {
    router.push(`/dashboard/employee/${employee.id}`);
  };

  return (
    <td
      className={`relative h-[55px] w-[140px] cursor-pointer border border-gray-200 text-center align-middle font-semibold transition select-none ${
        hasEmployee ? "bg-gray-100" : "bg-white text-gray-400"
      }`}
      onMouseEnter={() => setShowTooltip(true)}
      onMouseLeave={() => setShowTooltip(false)}
    >
      {hasEmployee ? (
        <div className="flex flex-col gap-1 overflow-hidden">
          {employees.map((e) => (
            <div
              key={e.id}
              className={`${getEmployeeColor(e)} truncate rounded px-1 text-sm text-white`}
              onClick={() => handleClick(e)}
            >
              {e.name}
            </div>
          ))}
        </div>
      ) : (
        <span className="text-sm text-gray-400">—</span>
      )}

      {showTooltip && hasEmployee && (
        <div className="absolute top-full left-1/2 z-50 mt-2 w-max max-w-[220px] -translate-x-1/2 transform rounded-lg bg-black p-3 text-sm whitespace-pre-line text-white shadow-xl">
          {employees.map((e) => (
            <div key={e.id} className="mb-1">
              <div className="font-bold">{e.name}</div>
              {e.phone && <div>{e.phone}</div>}
              {e.note && <div className="text-gray-300">{e.note}</div>}
            </div>
          ))}
        </div>
      )}
    </td>
  );
}
