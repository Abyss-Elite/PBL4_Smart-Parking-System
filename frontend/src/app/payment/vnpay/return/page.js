"use client";

import { useEffect, useState, useRef } from "react";
import { CheckCircle2, CreditCard, Landmark, Calendar, FileText, XCircle } from "lucide-react";
import { formatCurrency } from "@/utils/formatCurrency";
import { userBookingPageAPI } from "@/api/parking-lot/userBookingPageAPI";
import PATH from "@/routes/PATH";

export default function ReturnPage() {
  const [status, setStatus] = useState(null); // SUCCESS | FAILED
  const [data, setData] = useState(null);

  useEffect(() => {
    const params = new URLSearchParams(window.location.search);
    const txnRef = params.get("vnp_TxnRef");

    if (!txnRef) {
      setStatus("FAILED");
      return;
    }

    const dataKey = `vnpay_data_${txnRef}`;
    const statusKey = `vnpay_status_${txnRef}`;

    const cachedStatus = sessionStorage.getItem(statusKey);
    const cachedData = sessionStorage.getItem(dataKey);

    if (cachedStatus && cachedData) {
      setData(JSON.parse(cachedData));
      setStatus(cachedStatus);
      return;
    }

    if (window.__vnpay_processing) {
      const timer = setTimeout(() => {
        const s = sessionStorage.getItem(statusKey);
        const d = sessionStorage.getItem(dataKey);
        if (s && d) {
          setData(JSON.parse(d));
          setStatus(s);
        } else {
          setStatus("FAILED");
        }
      }, 300); // 300ms là đủ

      return () => clearTimeout(timer);
    }

    // 🚀 LẦN ĐẦU TIÊN GỌI API
    window.__vnpay_processing = true;

    const payload = {};
    params.forEach((v, k) => (payload[k] = v));

    (async () => {
      try {
        const res = await userBookingPageAPI.paymentReturn(payload);

        if (res.data?.status === "SUCCESS" && res.data?.data?.status === "SUCCESS") {
          sessionStorage.setItem(dataKey, JSON.stringify(res.data.data.details));
          sessionStorage.setItem(statusKey, "SUCCESS");
          setData(res.data.data.details);
          setStatus("SUCCESS");
        } else {
          sessionStorage.setItem(statusKey, "FAILED");
          setStatus("FAILED");
        }
      } catch {
        sessionStorage.setItem(statusKey, "FAILED");
        setStatus("FAILED");
      }
    })();
  }, []);

  if (!status) {
    return (
      <div className="flex min-h-screen items-center justify-center">
        <p className="text-gray-500">Đang xác minh thanh toán...</p>
      </div>
    );
  }

  if (status === "FAILED") {
    return (
      <div className="flex min-h-screen items-center justify-center bg-gray-100 p-4">
        <div className="w-full max-w-md rounded-xl bg-white p-6 text-center shadow-lg">
          <XCircle className="mx-auto h-14 w-14 text-red-500" />
          <h2 className="mt-3 text-xl font-bold text-gray-800">Thanh toán thất bại</h2>
          <p className="mt-2 text-sm text-gray-500">
            Giao dịch không hợp lệ hoặc đã được xử lý trước đó
          </p>
          <a
            href="/parkingReservation"
            className="mt-6 inline-block rounded-lg bg-blue-600 px-6 py-2 text-white hover:bg-blue-700"
          >
            Quay lại đặt chỗ
          </a>
        </div>
      </div>
    );
  }

  const amount = formatCurrency(Number(data.vnp_Amount));

  const formatDate = (date) => {
    if (!date) return "";
    return `${date.slice(6, 8)}/${date.slice(4, 6)}/${date.slice(
      0,
      4
    )} ${date.slice(8, 10)}:${date.slice(10, 12)}:${date.slice(12, 14)}`;
  };

  const items = [
    {
      label: "Số tiền",
      value: amount + " VNĐ",
      icon: <CreditCard className="h-5 w-5 text-blue-600" />,
    },
    {
      label: "Ngân hàng",
      value: data.vnp_BankCode,
      icon: <Landmark className="h-5 w-5 text-purple-600" />,
    },
    {
      label: "Mã giao dịch ngân hàng",
      value: data.vnp_BankTranNo,
      icon: <CreditCard className="h-5 w-5 text-green-600" />,
    },
    {
      label: "Nội dung đơn hàng",
      value: data.vnp_OrderInfo,
      icon: <FileText className="h-5 w-5 text-orange-600" />,
    },
    {
      label: "Thời gian thanh toán",
      value: formatDate(data.vnp_PayDate),
      icon: <Calendar className="h-5 w-5 text-red-600" />,
    },
  ];

  return (
    <div className="flex min-h-screen items-center justify-center bg-gray-100 p-4">
      <div className="w-full max-w-md rounded-xl bg-white p-6 shadow-lg">
        <div className="mb-5 flex flex-col items-center">
          <CheckCircle2 className="h-14 w-14 text-green-500" />
          <h2 className="mt-3 text-xl font-bold text-gray-800">Thanh toán thành công</h2>
          <p className="mt-1 text-center text-sm text-gray-500">Cảm ơn bạn đã sử dụng dịch vụ</p>
        </div>

        <div className="space-y-3">
          {items.map((item, idx) => (
            <div
              key={idx}
              className="flex items-center gap-3 rounded-lg border border-gray-200 bg-gray-50 p-3"
            >
              {item.icon}
              <div>
                <p className="text-sm text-gray-500">{item.label}</p>
                <p className="text-[15px] font-semibold text-gray-800">{item.value}</p>
              </div>
            </div>
          ))}
        </div>

        {localStorage.getItem("accessToken") ? (
          <div className="mt-6">
            <a
              href={PATH.DASHBOARD.ADMIN_HOME}
              className="block rounded-lg bg-blue-600 py-2 text-center font-medium text-white transition hover:bg-blue-700"
            >
              Trở về trang chính
            </a>
          </div>
        ) : (
          <div className="mt-6">
            <a
              href="/parkingReservation"
              className="block rounded-lg bg-blue-600 py-2 text-center font-medium text-white transition hover:bg-blue-700"
            >
              Trở về trang đặt chỗ
            </a>
          </div>
        )}

        <p className="mt-4 text-center text-xs text-gray-500">
          Backend đã xác minh chữ ký VNPay thành công
        </p>
      </div>
    </div>
  );
}
