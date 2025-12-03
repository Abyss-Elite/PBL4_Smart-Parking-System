"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { bookingAPI } from "@/api/booking/bookingAPI";
import { convertOldCarsToNew } from "@/utils/booking";
import { Car } from "@/types/car";
import PATH from "@/routes/PATH";

export default function ConfirmPage() {
  const [cars, setCars] = useState<Car[]>([]);
  const router = useRouter();
  const [name, setName] = useState("");
  const [phone, setPhone] = useState("");

  useEffect(() => {
    const data = localStorage.getItem("parking_booking");
    if (data) {
      try {
        const oldCars = JSON.parse(data);
        const newCars = convertOldCarsToNew(oldCars);
        setCars(newCars);
      } catch (err) {
        console.error("Lỗi parse data:", err);
      }
    }
  }, []);

  const handleContinue = async () => {
    if (!name || !phone) {
      alert("Vui lòng điền đầy đủ thông tin khách hàng");
      return;
    }
    try {
      const payload = {
        customerName: name,
        customerPhone: phone,
        cars: cars,
      };
      console.log(payload);
      const res = await bookingAPI.createBooking(payload);
      const booking = res.data;
      router.push(PATH.PARKING_RESERVATION.PENDINGBILL(booking.id));
    } catch (err) {
      console.error("Booking Error:", err);
      alert("Không thể tạo booking. Vui lòng thử lại.");
    }
  };

  return (
    <div className="mx-auto max-w-4xl space-y-8 px-6 py-8">
      <h1 className="text-3xl font-bold tracking-tight">Booking Summary</h1>

      <div className="rounded-2xl border bg-white p-6 shadow-sm">
        <h2 className="mb-4 text-xl font-semibold">Thông tin khách hàng</h2>

        <div className="grid gap-4 md:grid-cols-2">
          <div>
            <label className="mb-1 block text-sm font-medium">Họ và tên</label>
            <input
              type="text"
              placeholder="Nhập họ tên"
              value={name}
              onChange={(e) => setName(e.target.value)}
              className="w-full rounded-lg border px-3 py-2 shadow-sm focus:border-green-500 focus:outline-none"
            />
          </div>

          <div>
            <label className="mb-1 block text-sm font-medium">Số điện thoại</label>
            <input
              type="tel"
              placeholder="Nhập số điện thoại"
              value={phone}
              onChange={(e) => setPhone(e.target.value)}
              className="w-full rounded-lg border px-3 py-2 shadow-sm focus:border-green-500 focus:outline-none"
            />
          </div>
        </div>
      </div>

      <div className="rounded-2xl border bg-white p-6 shadow-sm">
        <h2 className="mb-4 text-xl font-semibold">Danh sách xe đã đặt</h2>

        {cars.length === 0 ? (
          <p className="text-gray-500">Không có dữ liệu nào.</p>
        ) : (
          <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-3">
            {cars.map((c, i) => (
              <div
                key={i}
                className="rounded-xl border bg-gray-50 p-5 shadow-sm transition hover:shadow-md"
              >
                <p className="mb-1">
                  <strong>Chỗ: </strong> {c.spotName}
                </p>
                <p className="mb-1">
                  <strong>Biển số: </strong> {c.licensePlate}
                </p>
                <p className="mb-1">
                  <strong>Chế độ: </strong> {c.mode}
                </p>

                {c.mode == "week" && (
                  <p className="text-sm text-gray-600">Tuần: {c.startTimeBooking}</p>
                )}
                {c.mode == "month" && (
                  <p className="text-sm text-gray-600">Tháng: {c.startTimeBooking}</p>
                )}
              </div>
            ))}
          </div>
        )}
      </div>

      <div>
        <button
          onClick={handleContinue}
          className="rounded-lg bg-green-600 px-6 py-2 text-white shadow-md transition hover:bg-green-700"
        >
          Continue
        </button>
      </div>
    </div>
  );
}
