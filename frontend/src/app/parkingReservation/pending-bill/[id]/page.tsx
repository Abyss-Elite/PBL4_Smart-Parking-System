"use client";

import React, { useEffect, useState } from "react";
import { bookingAPI } from "@/api/booking/bookingAPI";
import { useRouter } from "next/navigation";
import PATH from "@/routes/PATH";
import { CarBookingPending } from "@/types/car";
import { format } from "date-fns";

type PageProps = {
  params: Promise<{ id: string }>;
};

export default function PendingBillPage({ params }: PageProps) {
  const { id } = React.use(params);
  const router = useRouter();

  const [loading, setLoading] = useState(true);
  const [booking, setBooking] = useState<{
    customerName: string;
    customerPhone: string;
    cars: CarBookingPending[];
    totalAmount: number;
  }>({
    customerName: "",
    customerPhone: "",
    cars: [],
    totalAmount: 0,
  });

  useEffect(() => {
    const fetchData = async () => {
      try {
        const res = await bookingAPI.getBooking(id);
        setBooking(res.data);
        // setBooking(mockBooking);
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
      `${PATH.PARKING_RESERVATION.PAYMENT}?bookingId=${id}&amount=${booking.totalAmount}`
    );
  };

  if (loading) {
    return <div className="mx-auto mt-10 text-center text-gray-500">Đang tải hóa đơn...</div>;
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
                className="rounded-xl border bg-gray-50 p-4 shadow-sm transition hover:shadow-md"
              >
                <p>
                  <strong>Chỗ:</strong> {car.parkingSpot.name}
                </p>
                <p>
                  <strong>Biển số:</strong> {car.licensePlate}
                </p>

                <ul>
                  <strong>Thời gian đặt</strong>
                  <li className="ml-2.5 text-sm text-gray-600">
                    Bắt đầu: {format(new Date(car.startTimeBooking), "HH:mm dd-MM-yyyy")}
                  </li>
                  <li className="ml-2.5 text-sm text-gray-600">
                    Kết thúc: {format(new Date(car.endTimeBooking), "HH:mm dd-MM-yyyy")}
                  </li>
                </ul>
              </div>
            ))}
          </div>
        )}
      </div>

      <div className="rounded-2xl border bg-white p-6 shadow-sm">
        <div className="flex items-center justify-between text-xl font-semibold">
          <span>Tổng tiền:</span>
          <span className="text-green-600">{booking.totalAmount?.toLocaleString("vi-VN")}₫</span>
        </div>

        <button
          onClick={handlePayment}
          className="mt-6 w-full cursor-pointer rounded-lg bg-green-600 px-6 py-3 text-white shadow-md transition hover:bg-green-700"
        >
          Thanh toán
        </button>
      </div>
    </div>
  );
}
