"use client";

import { useState } from "react";

export default function BookingForm({ slot, bookingMode, onAddCar }) {
  const [plate, setPlate] = useState("");
  const [weekStart, setWeekStart] = useState("");
  const [month, setMonth] = useState("");
  const [error, setError] = useState("");

  const today = new Date();

  const validateFuture = () => {
    if (bookingMode === "week") {
      if (!weekStart) return "Chưa chọn tuần";
      if (new Date(weekStart) < today) return "Không thể đặt tuần trong quá khứ";
    } else {
      if (!month) return "Chưa chọn tháng";
      if (new Date(month + "-01") < today) return "Không thể đặt tháng trong quá khứ";
    }
    return null;
  };

  const handleAdd = () => {
    setError("");
    const err = validateFuture();
    if (err) return setError(err);
    if (!plate) return setError("Hãy nhập biển số xe");

    onAddCar({
      plate: plate.toUpperCase(),
      mode: bookingMode,
      weekStart: bookingMode === "week" ? weekStart : null,
      month: bookingMode === "month" ? month : null,
    });

    setPlate("");
    setWeekStart("");
    setMonth("");
  };

  return (
    <div className="mt-4 space-y-3 rounded-xl border bg-white p-4 shadow">
      <p className="font-semibold">Nhập thông tin xe cho chỗ {slot}</p>

      <input
        type="text"
        placeholder="Biển số xe"
        value={plate}
        onChange={(e) => setPlate(e.target.value)}
        className="w-full rounded border p-2"
      />

      {bookingMode === "week" ? (
        <input
          type="date"
          value={weekStart}
          onChange={(e) => setWeekStart(e.target.value)}
          className="w-full rounded border p-2"
        />
      ) : (
        <input
          type="month"
          value={month}
          onChange={(e) => setMonth(e.target.value)}
          className="w-full rounded border p-2"
        />
      )}

      {error && <p className="text-red-500">{error}</p>}

      <button onClick={handleAdd} className="w-full rounded-lg bg-blue-600 p-2 text-white cursor-pointer hover:bg-blue-700">
        Thêm xe
      </button>
    </div>
  );
}
