"use client";
import { useState } from "react";
import { addDays, format, lastDayOfMonth } from "date-fns";
import { formatLicensePlate } from "@/utils/formatPlate";

export default function BookingForm({ slot, bookingMode, onAddCar, booked, onSwitchToMonth }) {
  const [licensePlate, setLicensePlate] = useState("");
  const [startDate, setStartDate] = useState("");
  const [numWeeks, setNumWeeks] = useState(1);
  const [month, setMonth] = useState("");
  const [error, setError] = useState("");

  const validateLicensePlate = (plate) => {
    const regex = /^\d{2}[A-Z]-\d{4,5}$/;
    return regex.test(plate);
  };

  const validateNumWeeks = (n) => {
    return Number.isInteger(n) && n > 0;
  };

  const handleAdd = () => {
    setError("");

    if (!licensePlate.trim()) return setError("Hãy nhập biển số xe");
    if (!validateLicensePlate(licensePlate))
      return setError("Biển số không hợp lệ. Ví dụ: 29A-12345");

    let start, end;

    if (bookingMode === "week") {
      if (!startDate) return setError("Chọn ngày bắt đầu");

      const today = new Date();
      const selectedStart = new Date(startDate);

      // So sánh: nếu ngày bắt đầu < hôm nay => lỗi
      if (selectedStart < today.setHours(0, 0, 0, 0)) {
        return setError("Ngày bắt đầu không được ở quá khứ");
      }

      if (!validateNumWeeks(numWeeks)) return setError("Số tuần phải là số nguyên dương");

      start = startDate;
      end = format(addDays(selectedStart, numWeeks * 7 - 1), "yyyy-MM-dd");
    } else {
      if (!month) return setError("Chọn tháng");
      const [year, monthNum] = month.split("-").map(Number);
      const firstDay = new Date(year, monthNum - 1, 1);
      const lastDay = lastDayOfMonth(firstDay);
      start = format(firstDay, "yyyy-MM-dd");
      end = format(lastDay, "yyyy-MM-dd");
    }

    const success = onAddCar({
      licensePlate: formatLicensePlate(licensePlate),
      startTimeBooking: start,
      endTimeBooking: end,
      mode: bookingMode,
    });

    if (success === false) {
      setError("Thời gian bị trùng! Hãy chọn thời gian khác.");
      return;
    }

    // reset form
    setLicensePlate("");
    setStartDate("");
    setNumWeeks(1);
    setMonth("");
    setError("");
  };

  return (
    <div className="mt-4 space-y-3 rounded-xl border bg-white p-4 shadow">
      <p className="font-semibold">Nhập thông tin xe cho chỗ {slot}</p>

      {/* Biển số xe */}
      <div className="flex-col gap-0">
        <p>Nhập biển số xe</p>
        <input
          type="text"
          placeholder="Ví dụ: 29A-12345"
          value={licensePlate}
          onChange={(e) => setLicensePlate(e.target.value.toUpperCase())}
          className="w-full rounded border p-2"
        />
      </div>

      {/* Tuỳ theo mode */}
      {bookingMode === "week" ? (
        <>
          <div className="flex-col gap-0">
            <p>Nhập ngày bắt đầu bạn muốn đặt</p>
            {/* <input
              type="date"
              value={startDate}
              onChange={(e) => setStartDate(e.target.value)}
              className="w-full rounded border p-2"
            /> */}
            <input
              type="date"
              value={startDate}
              onChange={(e) => {
                const selected = e.target.value;
                setStartDate(selected);

                const today = new Date();
                today.setHours(0, 0, 0, 0);
                const selectedDate = new Date(selected);
                selectedDate.setHours(0, 0, 0, 0);

                if (selectedDate < today) {
                  setError("Ngày bắt đầu không được ở quá khứ");
                } else {
                  setError(""); // clear lỗi nếu chọn ngày hợp lệ
                }
              }}
              className="w-full rounded border p-2"
            />
          </div>

          <div className="flex-col gap-0">
            <p>Nhập số tuần</p>
            <input
              type="number"
              min={1}
              value={numWeeks}
              onChange={(e) => setNumWeeks(Number(e.target.value))}
              className="w-full rounded border p-2"
              placeholder="Số tuần"
            />
          </div>

          {numWeeks >= 4 && (
            <div className="mt-2 rounded-lg bg-yellow-100 p-2 text-sm text-yellow-800">
              Bạn đã chọn số tuần đủ để đặt theo tháng.
              <br />
              <button
                onClick={() => {
                  onSwitchToMonth();
                }}
                className="mt-1 cursor-pointer font-semibold text-blue-700 underline hover:text-blue-900"
              >
                Nhấn vào đây để chuyển sang đặt tháng
              </button>
            </div>
          )}

          {startDate && validateNumWeeks(numWeeks) && (
            <p>
              Ngày kết thúc: {format(addDays(new Date(startDate), numWeeks * 7 - 1), "yyyy-MM-dd")}
            </p>
          )}
        </>
      ) : (
        <>
          <div className="flex-col gap-0">
            <p>Chọn tháng</p>
            {/* <input
              type="month"
              value={month}
              onChange={(e) => setMonth(e.target.value)}
              className="w-full rounded border p-2"
            /> */}
            <input
              type="month"
              value={month}
              onChange={(e) => {
                const selected = e.target.value; // "2025-12"
                const [year, monthNum] = selected.split("-").map(Number);
                const today = new Date();
                const firstDayOfSelected = new Date(year, monthNum - 1, 1);

                // Tháng quá khứ
                if (
                  firstDayOfSelected.getFullYear() < today.getFullYear() ||
                  (firstDayOfSelected.getFullYear() === today.getFullYear() &&
                    firstDayOfSelected.getMonth() < today.getMonth())
                ) {
                  setError("Không được chọn tháng quá khứ");
                  setMonth("");
                  return;
                }

                // Nếu chọn tháng hiện tại và ngày hôm nay >= 10 → không cho đặt full tháng
                if (
                  firstDayOfSelected.getFullYear() === today.getFullYear() &&
                  firstDayOfSelected.getMonth() === today.getMonth() &&
                  today.getDate() >= 10
                ) {
                  setError("Không thể đặt full tháng hiện tại từ ngày 16 trở đi");
                  setMonth("");
                  return;
                }

                // Hợp lệ → xóa lỗi
                setError("");
                setMonth(selected);
              }}
              className="w-full rounded border p-2"
            />
          </div>

          {month &&
            (() => {
              const [year, monthNum] = month.split("-").map(Number);
              const firstDay = new Date(year, monthNum - 1, 1);
              const lastDay = lastDayOfMonth(firstDay);
              return (
                <p>
                  Ngày bắt đầu → kết thúc: {format(firstDay, "yyyy-MM-dd")} →{" "}
                  {format(lastDay, "yyyy-MM-dd")}
                </p>
              );
            })()}
        </>
      )}

      {error && <p className="text-sm text-red-500">{error}</p>}

      <button
        onClick={handleAdd}
        className="w-full cursor-pointer rounded-lg bg-blue-600 p-2 text-white hover:bg-blue-700"
      >
        Thêm xe
      </button>
    </div>
  );
}
