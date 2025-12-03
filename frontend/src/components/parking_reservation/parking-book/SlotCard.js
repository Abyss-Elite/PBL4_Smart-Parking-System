"use client";
import { ParkingCircle, CheckCircle2, Clock, Car } from "lucide-react";
import { format, parseISO } from "date-fns";

export default function SlotCard({
  slot,
  booked = [],
  bookingMode,
  userCars = [],
  onSelectCar,
  selected,
}) {
  // Kiểm tra xe user ở slot này
  const userCarCurrent = userCars.find((c) => selected === slot);

  const isBookedByOthers = booked.length > 0;

  let status = {
    border: "border-gray-300",
    color: "text-green-600",
    text: "Chưa ai đặt",
    icon: <ParkingCircle size={18} />,
  };

  if (selected === slot) {
    status = {
      border: "border-yellow-400",
      color: "text-yellow-600",
      text: "Bạn đang chọn chỗ này",
      icon: <ParkingCircle size={18} />,
    };
  } else if (userCarCurrent) {
    status = {
      border: "border-blue-600",
      color: "text-blue-600",
      text: "Xe bạn đang đặt",
      icon: <CheckCircle2 size={18} />,
    };
  } else if (isBookedByOthers) {
    status = {
      border: "border-orange-500",
      color: "text-orange-500",
      text: "Đã được đặt",
      icon: <Clock size={18} />,
    };
  }

  return (
    <div
      onClick={() => onSelectCar && onSelectCar(slot)}
      className={`cursor-pointer rounded-xl border-2 p-2 shadow hover:bg-gray-50 ${status.border}`}
    >
      <div className="flex items-center justify-between">
        <p className="text-lg font-semibold">{slot}</p>
        <span className={status.color}>{status.icon}</span>
      </div>
      <p className={`mt-1 text-sm ${status.color}`}>{status.text}</p>

      {booked.length > 0 && (
        <div className="mt-2 flex flex-wrap gap-1">
          {booked.map((b, idx) => (
            <span
              key={`${b.id}-${b.startTimeBooking}-${b.endTimeBooking}-${idx}`}
              className={`rounded px-2 py-0.5 text-xs ${
                userCars.some((c) => c.id === b.id)
                  ? "bg-blue-100 text-blue-800"
                  : "bg-red-100 text-red-800"
              }`}
            >
              {format(parseISO(b.startTimeBooking), "dd/MM/yyyy")} →{" "}
              {format(parseISO(b.endTimeBooking), "dd/MM/yyyy")}
            </span>
          ))}
        </div>
      )}
    </div>
  );
}
