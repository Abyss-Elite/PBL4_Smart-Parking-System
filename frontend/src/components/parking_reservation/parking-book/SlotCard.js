"use client";
import { ParkingCircle, CheckCircle2, Clock, Car } from "lucide-react";
import { format, isValid, parseISO } from "date-fns";

export default function SlotCard({
  slot,
  booked = { week: [], month: [] },
  bookingMode,
  currentWeek,
  currentMonth,
  userCars = [],
  onSelectCar,
  selected,
}) {
  const currentTime = bookingMode === "week" ? currentWeek : currentMonth;

  // Chuẩn hóa string cho check
  let currentTimeStr = "";
  let currentMonthStr = "";

  if (bookingMode === "week" && currentTime) {
    const d = parseISO(currentTime);
    currentTimeStr = isValid(d) ? format(d, "yyyy-MM-dd") : "";
    currentMonthStr = currentTimeStr.slice(0, 7); // YYYY-MM
  } else {
    currentTimeStr = currentTime || "";
    currentMonthStr = currentTimeStr;
  }

  // Xe của user cùng slot
  const userCarCurrent = userCars.find((c) =>
    bookingMode === "week" ? c.weekStart === currentTimeStr : c.month === currentTimeStr
  );

  const userCarOther = userCars.find((c) =>
    bookingMode === "week" ? c.weekStart !== currentTimeStr : c.month !== currentTimeStr
  );

  // Check chéo tuần ↔ tháng với booked của người khác
  const isBookedByOthers =
    (!userCarCurrent &&
      ((bookingMode === "week" &&
        (booked.week.includes(currentTimeStr) ||
          booked.month.some((m) => currentTimeStr.startsWith(m)))) ||
        (bookingMode === "month" &&
          (booked.month.includes(currentMonthStr) ||
            booked.week.some((w) => w.startsWith(currentMonthStr)))))) ||
    false;

  // Logic hiển thị
  let status = {
    border: "border-gray-300",
    color: "text-green-600",
    text: "Chưa ai đặt",
    icon: <ParkingCircle size={18} />,
    clickable: true,
  };

  if (userCarCurrent) {
    status = {
      border: "border-blue-600",
      color: "text-blue-600",
      text: "Xe bạn đang đặt",
      icon: <CheckCircle2 size={18} />,
      clickable: true,
    };
  } else if (userCarOther) {
    status = {
      border: "border-blue-400",
      color: "text-blue-400",
      text: "Xe bạn đã thêm vào danh sách đặt (khác thời gian)",
      icon: <Car size={18} />,
      clickable: true,
    };
  } else if (isBookedByOthers) {
    status = {
      border: "border-orange-500",
      color: "text-orange-500",
      text: "Đã được đặt (trùng thời gian)",
      icon: <Clock size={18} />,
      clickable: true, // giữ clickable nếu muốn check lỗi khi thêm xe
    };
  } else if (selected === slot) {
    status = {
      border: "border-yellow-400",
      color: "text-yellow-600",
      text: "Bạn đang chọn chỗ này",
      icon: <ParkingCircle size={18} />,
      clickable: true,
    };
  }

  const bookedTimesDisplay = [...booked.week, ...booked.month];

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

      {booked.week.length > 0 && (
        <div className="mt-2 space-y-1">
          {booked.week.map((b, i) => (
            <p key={`w-${i}`} className="text-xs text-red-500">
              ● Tuần: {b}
            </p>
          ))}
        </div>
      )}

      {booked.month.length > 0 && (
        <div className="mt-1 space-y-1">
          {booked.month.map((b, i) => (
            <p key={`m-${i}`} className="text-xs text-red-500">
              ● Tháng: {b}
            </p>
          ))}
        </div>
      )}
    </div>
  );
}
