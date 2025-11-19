"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";

export default function ConfirmPage() {
  const [cars, setCars] = useState([]);
  const router = useRouter();
  const [name, setName] = useState("");
  const [phone, setPhone] = useState("");

  useEffect(() => {
    const data = localStorage.getItem("parking_booking");
    if (data) {
      setCars(JSON.parse(data));
    }
  }, []);

  const handleContinue = () => {
    if (!name || !phone) {
      alert("Vui lòng điền đầy đủ thông tin khách hàng");
      return;
    }

    setCustomerInfo({ name, phone });

    console.log("Booking data:", { customerInfo: { name, phone }, vehicleForms });

    router.push(PATH.DASHBOARD.PARKING_LOT_MANAGEMENT.RESERVATION.PAYMENT);
  };

  return (
    <div className="container mx-auto flex flex-col gap-6 px-4 py-6">
      <h1 className="text-2xl font-bold">Booking Summary</h1>

      <div className="flex flex-col gap-4 rounded border p-4 shadow">
        <h2 className="text-lg font-semibold">Thông tin khách hàng</h2>
        <input
          type="text"
          placeholder="Họ và tên"
          value={name}
          onChange={(e) => setName(e.target.value)}
          className="w-full rounded border px-2 py-1"
        />
        <input
          type="tel"
          placeholder="Số điện thoại"
          value={phone}
          onChange={(e) => setPhone(e.target.value)}
          className="w-full rounded border px-2 py-1"
        />
      </div>

      <div className="flex flex-col gap-4">
        <h2 className="text-lg font-semibold">Xe đã đặt 2</h2>
        <div className="grid grid-cols-1 gap-4 md:grid-cols-2 lg:grid-cols-3">
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
      </div>

      <button
        onClick={handleContinue}
        className="self-start rounded bg-green-600 px-4 py-2 text-white hover:bg-green-700"
      >
        Continue
      </button>
    </div>
  );
}
