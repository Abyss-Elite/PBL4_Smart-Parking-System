"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
export default function ShiftCell({ employee }) {
  const [showTooltip, setShowTooltip] = useState(false);
  const router = useRouter();
  const hasEmployee = !!employee;

  const handleClick = () => {
    if (hasEmployee) {
      router.push(`/dashboard/employee/${employee.id}`);
    }
  };

  return (
    <td
      className={`relative h-[55px] w-[140px] cursor-pointer border border-gray-200 text-center align-middle font-semibold transition select-none ${
        hasEmployee ? employee.color + " text-white" : "bg-white text-gray-500"
      }`}
      onMouseEnter={() => setShowTooltip(true)}
      onMouseLeave={() => setShowTooltip(false)}
      onClick={handleClick}
    >
      {hasEmployee ? (
        <div className="truncate">{employee.name}</div>
      ) : (
        <span className="text-sm text-gray-400">—</span>
      )}

      {showTooltip && hasEmployee && (
        <div className="absolute top-full left-1/2 z-50 mt-2 w-max max-w-[220px] -translate-x-1/2 transform rounded-lg bg-black p-3 text-sm whitespace-pre-line text-white shadow-xl">
          <div className="font-bold">{employee.name}</div>
          {employee.phone && <div>{employee.phone}</div>}
          {employee.note && <div className="text-gray-300">{employee.note}</div>}
        </div>
      )}
    </td>
  );
}
