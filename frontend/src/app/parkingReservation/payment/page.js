"use client";

import { useSearchParams } from "next/navigation";
import { useEffect, useState } from "react";
import { userBookingPageAPI } from "@/api/parking-lot/userBookingPageAPI";

export default function PaymentPage() {
  const searchParams = useSearchParams();

  const id = searchParams.get("id");
  const total = searchParams.get("total");

  const [loading, setLoading] = useState(false);

  useEffect(() => {
    const paymentData = async () => {
      setLoading(true);
      try {
        const res = await userBookingPageAPI.vnpayPayment({ amount: total, orderId: id });
        const data = res.data;
        if (data.paymentUrl) {
          window.location.href = data.paymentUrl;
        } else {
          alert("Không nhận được paymentUrl");
        }
      } catch (err) {
        console.error(err);
        alert("Lỗi tạo thanh toán");
      } finally {
        setLoading(false);
      }
    };
    paymentData();
  }, [id, total]);

  return (
    <div className="h-screen w-screen content-center">
      {loading && (
        <p className="text-2xl font-medium text-green-400 shadow-emerald-400">Đang tạo...</p>
      )}
    </div>
  );
}
