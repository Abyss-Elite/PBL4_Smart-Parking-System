import { useEffect, useState } from "react";

export default function BookingDetail({ bookingId, onClose }) {
  const data1 = {
    customer_name: "Thanh Tuyen",
    license_plate: "43H1-355.33",
    code: "2712",
    start_time:"2025-11-10T09:00:00Z",
    end_time:"2025-11-10T11:00:00Z",
    amount: 50000,
  };

  const [bookingDetail, setBookingDetail] = useState({
    customer_name: "",
    license_plate: "",
    code: "",
    start_time: "",
    end_time: "",
    amount: 0,
  });

  useEffect(() => {
    const fetchData = async () => {
      // const res = await carAPI.getBookingDetail(bookingId);
      // setBookingDetail(res.data);
      setBookingDetail(data1);
    };
    fetchData();
  }, [bookingId]);

  return (
    <>
      <div
        className="fixed inset-0 bg-black/50 backdrop-blur-sm z-40"
        onClick={onClose}
      ></div>

      <div
        className="
          fixed z-50 top-1/2 left-1/2 
          -translate-x-1/2 -translate-y-1/2
          bg-white rounded-xl shadow-lg p-6 w-[380px]
        "
      >
        <h2 className="text-xl font-semibold mb-4">
          Booking #{bookingDetail.code}
        </h2>

        <p><b>Customer:</b> {bookingDetail.customer_name}</p>
        <p><b>License:</b> {bookingDetail.license_plate}</p>
        <p><b>Code:</b> {bookingDetail.code}</p>
        <p><b>Start time:</b> {
          new Date(bookingDetail.start_time).toLocaleTimeString("vi-VN", {
            hour: "2-digit",
            minute: "2-digit",
            hour12: false,
            timeZone: "UTC"
          })
        }</p>
        <p><b>End time:</b> {
          new Date(bookingDetail.end_time).toLocaleTimeString("vi-VN", {
            hour: "2-digit",
            minute: "2-digit",
            hour12: false,
            timeZone: "UTC"
          })
        }</p>
        <p><b>Amount:</b> {bookingDetail.amount}</p>

        <button
          className="
            mt-5 px-4 py-2 
            bg-blue-600 text-white 
            rounded-md hover:bg-blue-700 w-full
          "
          onClick={onClose}
        >
          Close
        </button>
      </div>
    </>
  );
}
