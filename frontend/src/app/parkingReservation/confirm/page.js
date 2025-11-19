"use client";

import { useEffect, useState } from "react";

export default function ConfirmPage() {
  const [cars, setCars] = useState([]);

  useEffect(() => {
    const data = localStorage.getItem("parking_booking");
    if (data) {
      setCars(JSON.parse(data));
    }
  }, []);

  return (
    <div className="mx-auto mt-10 max-w-3xl rounded-xl bg-white p-6 shadow">
      <h1 className="mb-4 text-2xl font-bold">Xác nhận đặt xe</h1>

      {cars.length === 0 ? (
        <p className="text-gray-500">Không có dữ liệu nào.</p>
      ) : (
        <div className="space-y-4">
          {cars.map((c, i) => (
            <div key={i} className="rounded-lg border bg-gray-50 p-4">
              <p>
                <strong>Chỗ:</strong> {c.slot}
              </p>
              <p>
                <strong>Biển số:</strong> {c.plate}
              </p>
              <p>
                <strong>Giờ vào:</strong> {c.start}
              </p>
              <p>
                <strong>Giờ ra:</strong> {c.end}
              </p>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
