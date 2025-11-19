"use client";

import { useState } from "react";
import { validatePlate } from "@/utils/plateValidator";
import { isTimeOverlap } from "@/utils/timeUtils";
import BookedTimes from "./BookedTimes";
import SelectedCars from "./SelectedCars";

export default function BookingForm({ slot, bookedTimes, selectedCars, onAddCar, onClickSubmit }) {
  const [start, setStart] = useState("");
  const [end, setEnd] = useState("");
  const [plate, setPlate] = useState("");
  const [error, setError] = useState("");

  const handleAdd = () => {
    setError("");

    if (!validatePlate(plate)) {
      return setError("Biển số không hợp lệ. VD: 43A12345 hoặc 43A-12345");
    }

    if (!start || !end || start >= end) {
      return setError("Giờ vào phải nhỏ hơn giờ ra.");
    }

    const hasConflict = bookedTimes.some((b) => isTimeOverlap(start, end, b.start, b.end));

    if (hasConflict) {
      return setError("Khung giờ này đã có người đặt!");
    }

    onAddCar({
      slot,
      plate,
      start,
      end,
    });

    setPlate("");
    setStart("");
    setEnd("");
  };

  return (
    <div className="space-y-6 rounded-xl bg-white p-6 shadow">
      <div className="flex justify-between">
        <h2 className="text-xl font-semibold">
          Đặt chỗ cho <span className="text-blue-600">{slot}</span>
        </h2>
        <button
          onClick={onClickSubmit}
          className="rounded-lg bg-blue-600 px-6 py-3 font-semibold text-white shadow transition hover:bg-blue-700"
        >
          Đặt xe
        </button>
      </div>

      <BookedTimes booked={bookedTimes} />

      <div className="grid grid-cols-2 gap-4">
        <div>
          <p className="mb-1 text-sm font-medium">Giờ vào</p>
          <input
            type="time"
            className="w-full rounded-lg border p-2"
            value={start}
            onChange={(e) => setStart(e.target.value)}
          />
        </div>

        <div>
          <p className="mb-1 text-sm font-medium">Giờ ra</p>
          <input
            type="time"
            className="w-full rounded-lg border p-2"
            value={end}
            onChange={(e) => setEnd(e.target.value)}
          />
        </div>
      </div>

      <div>
        <p className="mb-1 text-sm font-medium">Biển số xe</p>
        <input
          type="text"
          placeholder="VD: 43A-12345"
          className="w-full rounded-lg border p-2"
          value={plate}
          onChange={(e) => setPlate(e.target.value.toUpperCase())}
        />
      </div>

      {error && <p className="text-sm text-red-500">{error}</p>}

      <button onClick={handleAdd} className="w-full rounded-lg bg-black p-3 font-medium text-white">
        Thêm xe này
      </button>

      <SelectedCars cars={selectedCars} />
    </div>
  );
}
