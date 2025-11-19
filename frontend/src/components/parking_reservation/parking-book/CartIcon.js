"use client";

import { useState, useEffect } from "react";
import { Car } from "lucide-react";

export default function CartIcon({ cars, onRemoveCar, onUpdateCar, onSubmit }) {
  const [showCart, setShowCart] = useState(false);
  const [localCars, setLocalCars] = useState(cars);

  useEffect(() => {
    setLocalCars(cars);
  }, [cars]);

  const handleChange = (idx, field, value) => {
    setLocalCars((prev) => prev.map((c, i) => (i === idx ? { ...c, [field]: value } : c)));
  };

  const handleOk = () => {
    onUpdateCar(localCars);
    setShowCart(false);
  };

  return (
    <div className="relative inline-block">
      <button
        onClick={() => setShowCart(!showCart)}
        className="relative rounded-full bg-blue-600 p-3 text-white shadow transition hover:bg-blue-700"
      >
        <Car size={20} />
        {cars.length > 0 && (
          <span className="absolute -top-1 -right-1 flex h-5 w-5 items-center justify-center rounded-full bg-red-500 text-xs font-semibold text-white">
            {cars.length}
          </span>
        )}
      </button>

      {showCart && (
        <div className="absolute right-0 z-50 mt-2 w-96 rounded-xl bg-white p-4 shadow-lg">
          <h3 className="mb-2 font-semibold text-gray-700">Xe đang đặt</h3>
          {localCars.length === 0 ? (
            <p className="text-sm text-gray-500">Chưa chọn xe nào</p>
          ) : (
            <div className="max-h-64 space-y-2 overflow-y-auto">
              {localCars.map((c, idx) => (
                <div key={idx} className="flex flex-col rounded-lg bg-gray-50 p-2 shadow-sm">
                  <div className="mb-1 flex items-center justify-between">
                    <p className="font-medium">
                      {c.slot} — {c.plate}
                    </p>
                    <button
                      onClick={() => onRemoveCar(idx)}
                      className="text-red-500 hover:text-red-700"
                    >
                      Xóa
                    </button>
                  </div>
                  <div className="grid grid-cols-2 gap-2">
                    <input
                      type="time"
                      value={c.start}
                      onChange={(e) => handleChange(idx, "start", e.target.value)}
                      className="w-full rounded border p-1 text-sm"
                    />
                    <input
                      type="time"
                      value={c.end}
                      onChange={(e) => handleChange(idx, "end", e.target.value)}
                      className="w-full rounded border p-1 text-sm"
                    />
                  </div>
                </div>
              ))}
            </div>
          )}
          <div className="mt-3 flex gap-2">
            <button
              onClick={handleOk}
              className="flex-1 rounded-lg bg-green-600 p-2 font-semibold text-white transition hover:bg-green-700"
            >
              OK
            </button>
            <button
              onClick={onSubmit}
              className="flex-1 rounded-lg bg-blue-600 p-2 font-semibold text-white transition hover:bg-blue-700"
            >
              Đặt tất cả
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
