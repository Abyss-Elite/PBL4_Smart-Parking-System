"use client";

import { ParkingCircle, Clock, CheckCircle2 } from "lucide-react";

export default function SlotCard({ slot, selected, bookedTimes = [], onClick }) {
  const isFree = bookedTimes.length === 0;
  const isPartial = bookedTimes.length > 0;

  return (
    <div
      onClick={onClick}
      className={`cursor-pointer rounded-xl border bg-white p-4 shadow-sm transition-all hover:shadow-md ${selected ? "border-blue-600 bg-blue-50" : "border-gray-300"} `}
    >
      <div className="flex items-center justify-between">
        <p className="text-lg font-semibold">{slot}</p>

        {selected ? (
          <CheckCircle2 size={18} className="text-blue-600" />
        ) : isFree ? (
          <ParkingCircle size={18} className="text-green-500" />
        ) : (
          <Clock size={18} className="text-orange-500" />
        )}
      </div>

      <p
        className={`mt-1 text-sm ${
          selected ? "text-blue-700" : isFree ? "text-green-600" : "text-orange-600"
        } `}
      >
        {selected
          ? "Đang chọn"
          : isFree
            ? "Chưa ai đặt"
            : `Đã có ${bookedTimes.length} khung giờ được đặt`}
      </p>
    </div>
  );
}
