import { useEffect, useState } from "react";
import { carAPI } from "@/api/car/carAPI";

export default function BookingDetail({ bookingId, spotId, date, onClose }) {
  // const data1 = {
  //   customerName: "Thanh Tuyen",
  //   cars: [
  //     {
  //       licensePlate: "43H-35533",
  //       fee: 150000,
  //     }
  //   ],
  //   code: "2712",
  // };
  const [bookingDetail, setBookingDetail] = useState({
    customerName: "",
    cars: [
      {
        licensePlate: "",
        fee: 0
      }
    ],
    code: ""
  });

  useEffect(() => {
    if (!bookingId || !spotId || !date) return;
    const fetchData = async () => {
      const res = await carAPI.getBookingDetail(bookingId, spotId, date);
      setBookingDetail(res.data);
      // setBookingDetail(data1);
    };
    fetchData();
  }, [bookingId, spotId, date]);

  console.log("BookingDetail render", { bookingId, spotId, date });
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

        <p><b>Customer:</b> {bookingDetail.customerName}</p>
        <p><b>License:</b> {bookingDetail.cars[0].licensePlate}</p>
        <p><b>Code:</b> {bookingDetail.code}</p>
        <p><b>Fee:</b> {bookingDetail.cars[0].fee}</p>

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
