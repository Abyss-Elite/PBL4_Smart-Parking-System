"use client";

import React, { useEffect, useState } from "react";
import { bookingAPI } from "@/api/booking/bookingAPI";
import { useRouter } from "next/navigation";
import PATH from "@/routes/PATH";

export default function PendingBillPage({ params }: { params: { id: string } }) {
  const { id } = params;
  const router = useRouter();
  const mockBooking = {
    customerName: "Nguyễn Văn A",
    customerPhone: "0905123456",
    total: 850000,
    cars: [
      {
        slot: "A12",
        plate: "43A-123.45",
        mode: "Theo tuần",
        weekStart: "Tuần 48 (25/11 - 01/12)",
      },
      {
        slot: "B03",
        plate: "43C-567.89",
        mode: "Theo tháng",
        month: "Tháng 12/2025",
      },
    ],
  };
  const [loading, setLoading] = useState(true);
  const [booking, setBooking] = useState<{
    customerName: string;
    customerPhone: string;
    cars: any[];
    total: number;
  }>({
    customerName: "",
    customerPhone: "",
    cars: [],
    total: 0,
  });

  useEffect(() => {
    const fetchData = async () => {
      try {
        // const res = await bookingAPI.getBooking(id);
        // setBooking(res.data);
        setBooking(mockBooking);
      } catch (err) {
        console.error("Error fetching bill:", err);
      } finally {
        setLoading(false);
      }
    };
    fetchData();
  }, [id]);

  const handlePayment = () => {
    router.push(
      `${PATH.PARKING_RESERVATION.PAYMENT}?id=${id}&total=${booking.total}`
    );

  };

  if (loading) {
    return (
      <div className="mx-auto mt-10 text-center text-gray-500">
        Đang tải hóa đơn...
      </div>
    );
  }

  return (
    <div className="mx-auto max-w-3xl space-y-6 px-6 py-8">
      <h1 className="text-3xl font-bold tracking-tight">Hóa đơn chờ</h1>

      <div className="rounded-2xl border bg-white p-6 shadow-sm">
        <h2 className="mb-4 text-xl font-semibold">Thông tin khách hàng</h2>

        <div className="space-y-2 text-gray-700">
          <p>
            <strong>Họ tên:</strong> {booking.customerName}
          </p>
          <p>
            <strong>Số điện thoại:</strong> {booking.customerPhone}
          </p>
        </div>
      </div>

      <div className="rounded-2xl border bg-white p-6 shadow-sm">
        <h2 className="mb-4 text-xl font-semibold">Xe đã đặt</h2>

        {booking.cars.length === 0 ? (
          <p className="text-gray-500">Không có dữ liệu xe.</p>
        ) : (
          <div className="space-y-4">
            {booking.cars.map((car, i) => (
              <div
                key={i}
                className="rounded-xl border bg-gray-50 p-4 shadow-sm hover:shadow-md transition"
              >
                <p>
                  <strong>Chỗ:</strong> {car.slot}
                </p>
                <p>
                  <strong>Biển số:</strong> {car.plate}
                </p>
                <p>
                  <strong>Chế độ:</strong> {car.mode}
                </p>

                {car.weekStart && (
                  <p className="text-sm text-gray-600">
                    Tuần: {car.weekStart}
                  </p>
                )}
                {car.month && (
                  <p className="text-sm text-gray-600">
                    Tháng: {car.month}
                  </p>
                )}
              </div>
            ))}
          </div>
        )}
      </div>

      <div className="rounded-2xl border bg-white p-6 shadow-sm">
        <div className="flex items-center justify-between text-xl font-semibold">
          <span>Tổng tiền:</span>
          <span className="text-green-600">
            {booking.total?.toLocaleString("vi-VN")}₫
          </span>
        </div>

        <button
          onClick={handlePayment}
          className="mt-6 w-full rounded-lg bg-green-600 px-6 py-3 text-white shadow-md hover:bg-green-700 transition"
        >
          Thanh toán
        </button>
      </div>
    </div>
  );
}
