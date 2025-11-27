"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { format } from "date-fns";
import BookingForm from "@/components/parking_reservation/parking-book/BookingForm";
import SlotList from "@/components/parking_reservation/parking-book/SlotList";
import CartIcon from "@/components/parking_reservation/parking-book/CartIcon";
import { Car } from "lucide-react";

export default function SlotPage() {
  const router = useRouter();
  const [selectedSlot, setSelectedSlot] = useState(null);
  const [cars, setCars] = useState([]);
  const [bookingMode, setBookingMode] = useState("week");

  const [weekStart, setWeekStart] = useState(format(new Date(), "yyyy-MM-dd"));
  const [month, setMonth] = useState(format(new Date(), "yyyy-MM"));

  const [errorMsg, setErrorMsg] = useState("");

  const slots = Array.from({ length: 50 }, (_, i) => `A${i + 1}`);

  const booked = {
    A1: { week: ["2025-11-24", "2025-12-01"], month: [] },
    A2: { week: [], month: ["2025-11"] },
    A3: { week: ["2025-11-24"], month: [] },
    A4: { week: [], month: ["2025-12"] },
  };

  const handleAddCar = (car) => {
    setErrorMsg("");

    if (!selectedSlot) {
      return setErrorMsg("Vui lòng chọn chỗ đậu xe trước khi thêm xe.");
    }

    const time = car.mode === "week" ? car.weekStart : car.month;
    if (!time) return setErrorMsg("Vui lòng chọn thời gian cho xe.");

    const slotBookedTimes = booked[selectedSlot] || { week: [], month: [] };

    if (car.mode === "week") {
      // Tuần trùng với booked tuần hoặc nằm trong booked tháng
      if (
        slotBookedTimes.week.includes(time) ||
        slotBookedTimes.month.some((m) => time.startsWith(m))
      ) {
        return setErrorMsg(`Chỗ ${selectedSlot} đã có người đặt tuần ${time}.`);
      }
    } else {
      // Tháng trùng với booked tháng hoặc có tuần nào trong booked tuần trùng tháng
      if (
        slotBookedTimes.month.includes(time) ||
        slotBookedTimes.week.some((w) => w.startsWith(time))
      ) {
        return setErrorMsg(`Chỗ ${selectedSlot} đã có người đặt tháng ${time}.`);
      }
    }

    // Kiểm tra trùng với xe đã thêm của user
    const userDuplicate = cars.find(
      (c) =>
        c.slot === selectedSlot &&
        c.mode === car.mode &&
        ((car.mode === "week" && c.weekStart === car.weekStart) ||
          (car.mode === "month" && c.month === car.month))
    );

    if (userDuplicate) {
      return setErrorMsg(`Bạn đã chọn chỗ ${selectedSlot} vào thời gian này.`);
    }

    // Thêm xe
    setCars((prev) => [...prev, { ...car, slot: selectedSlot }]);
  };

  const handleRemoveCar = (index) => setCars((prev) => prev.filter((_, i) => i !== index));

  return (
    <div className="grid grid-cols-12 gap-6 p-6">
      <div className="col-span-4">
        <h2 className="mb-2 flex items-center gap-2 text-xl font-bold">
          <Car size={22} className="text-blue-600" />
          Danh sách chỗ đậu
        </h2>

        <SlotList
          slots={slots}
          selectedSlot={selectedSlot}
          cars={cars}
          booked={booked}
          bookingMode={bookingMode}
          weekStart={weekStart}
          month={month}
          onSelectSlot={setSelectedSlot}
          onRemoveCar={handleRemoveCar}
        />
      </div>

      <div className="col-span-8 flex flex-col gap-6">
        <div className="flex items-center justify-between">
          <div className="flex gap-3">
            <button
              className={`cursor-pointer rounded border px-3 py-2 ${
                bookingMode === "week"
                  ? "bg-blue-600 text-white hover:bg-blue-700"
                  : "bg-white hover:bg-gray-50"
              }`}
              onClick={() => setBookingMode("week")}
            >
              Theo tuần
            </button>
            <button
              className={`cursor-pointer rounded border px-3 py-2 ${
                bookingMode === "month"
                  ? "bg-blue-600 text-white hover:bg-blue-700"
                  : "bg-white hover:bg-gray-50"
              }`}
              onClick={() => setBookingMode("month")}
            >
              Theo tháng
            </button>
          </div>

          <CartIcon cars={cars} onRemoveCar={handleRemoveCar} />
        </div>

        {errorMsg && <p className="font-medium text-red-500">{errorMsg}</p>}

        {selectedSlot ? (
          <BookingForm slot={selectedSlot} bookingMode={bookingMode} onAddCar={handleAddCar} />
        ) : (
          <div className="flex h-full items-center justify-center text-gray-500">
            <div className="text-center">
              <Car size={40} className="mx-auto mb-3 text-gray-400" />
              <p className="text-lg">Chọn 1 chỗ để bắt đầu đặt xe</p>
              <p className="mt-1 text-sm text-gray-400">
                Bãi có {slots.length} vị trí đang hoạt động
              </p>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
