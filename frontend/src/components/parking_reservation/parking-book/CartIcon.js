"use client";
import { Car } from "lucide-react";
import { useRouter } from "next/navigation";
import { useState } from "react";

export default function CartIcon({ cars, onRemoveCar }) {
  const router = useRouter();
  const [show, setShow] = useState(false);

  const submitAll = () => {
    if (cars.length === 0) return alert("Chưa có xe nào để đặt!");
    localStorage.setItem("parking_booking", JSON.stringify(cars));
    router.push("/parkingReservation/confirm");
  };

  return (
    <div className="relative inline-block">
      <button
        onClick={() => setShow(!show)}
        className="relative cursor-pointer rounded-full bg-blue-600 p-3 text-white shadow hover:bg-blue-700"
      >
        <Car size={20} />
        {cars.length > 0 && (
          <span className="absolute -top-1 -right-1 flex h-5 w-5 items-center justify-center rounded-full bg-red-500 text-xs text-white">
            {cars.length}
          </span>
        )}
      </button>

      {show && (
        <div className="absolute right-0 z-50 mt-2 w-96 rounded-xl bg-white p-4 shadow-lg">
          <p className="font-semibold">Xe đang đặt</p>

          {cars.length === 0 ? (
            <p className="text-sm text-gray-500">Chưa chọn xe nào</p>
          ) : (
            <div className="max-h-64 space-y-2 overflow-y-auto">
              {cars.map((c, i) => (
                <div key={c.id} className="flex justify-between rounded bg-gray-50 p-2 shadow">
                  <div>
                    <p className="font-semibold">{c.licensePlate}</p>
                    <p className="text-xs">{c.spotName}</p>
                    <p className="text-xs text-gray-500">
                      {c.startTimeBooking} → {c.endTimeBooking}
                    </p>
                  </div>
                  <button
                    onClick={() => onRemoveCar(i)}
                    className="text-red-500 hover:text-red-700"
                  >
                    Xóa
                  </button>
                </div>
              ))}
            </div>
          )}

          <button
            onClick={submitAll}
            className="mt-3 w-full cursor-pointer rounded-lg bg-green-600 p-2 text-white hover:bg-green-700"
          >
            Đặt tất cả
          </button>
        </div>
      )}
    </div>
  );
}
