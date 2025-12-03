"use client";
import { useEffect, useState } from "react";
import { differenceInDays } from "date-fns";
import BookingForm from "@/components/parking_reservation/parking-book/BookingForm";
import SlotList from "@/components/parking_reservation/parking-book/SlotList";
import CartIcon from "@/components/parking_reservation/parking-book/CartIcon";
import { carAPI } from "@/api/car/carAPI";
import { convertBookings } from "@/utils/convertBookings";
import { Car } from "lucide-react";

export default function SlotPage() {
  const [selectedSlot, setSelectedSlot] = useState(null);
  const [cars, setCars] = useState([]); // danh sách xe user thêm
  const [bookingMode, setBookingMode] = useState("week");
  const [parkingSpots, setParkingSpots] = useState();
  const [dataSlot, setDataSlot] = useState();
  const [booked, setBooked] = useState({});

  useEffect(() => {
    const fetchData = async () => {
      const res = await carAPI.getAllBooking();
      const res1 = await carAPI.getAllSpotReservation();
      const data = res.data;
      const data1 = res1.data;
      const parkingData = convertBookings(data);
      console.log("data: ", parkingData);
      setParkingSpots(parkingData);
      setDataSlot(data1);
    };
    fetchData();
  }, []);

  const slots = dataSlot?.map((s) => s.name) || [];

  // useEffect(() => {
  //   if (!parkingSpots) return;

  //   const initial = {};
  //   parkingSpots.forEach((s) => {
  //     initial[s.name] = [];
  //     if (s.startTimeBooking && s.endTimeBooking) {
  //       initial[s.name].push({
  //         id: s.id,
  //         spotId: s.id,
  //         licensePlate: null,
  //         startTimeBooking: s.startTimeBooking,
  //         endTimeBooking: s.endTimeBooking,
  //         mode:
  //           differenceInDays(new Date(s.endTimeBooking), new Date(s.startTimeBooking)) <= 14
  //             ? "week"
  //             : "month",
  //       });
  //     }
  //   });
  //   setBooked(initial);
  // }, [parkingSpots]);
  useEffect(() => {
    if (!parkingSpots) return;

    const initial = {};

    parkingSpots.forEach((s) => {
      if (!initial[s.name]) initial[s.name] = [];

      if (s.startTimeBooking && s.endTimeBooking) {
        initial[s.name].push({
          id: s.id,
          spotId: s.id,
          licensePlate: null,
          startTimeBooking: s.startTimeBooking,
          endTimeBooking: s.endTimeBooking,
          mode:
            differenceInDays(new Date(s.endTimeBooking), new Date(s.startTimeBooking)) <= 14
              ? "week"
              : "month",
        });
      }
    });

    setBooked(initial);
  }, [parkingSpots]);

  // const parkingSpots = [
  //   { id: 1, name: "B1", startTimeBooking: "2026-06-01", endTimeBooking: "2026-06-07" },
  //   { id: 2, name: "B2", startTimeBooking: "2026-06-10", endTimeBooking: "2026-06-23" },
  //   { id: 3, name: "B3", startTimeBooking: "2026-11-01", endTimeBooking: "2026-11-30" },
  //   { id: 4, name: "B4", startTimeBooking: "2026-12-01", endTimeBooking: "2026-12-31" },
  //   { id: 5, name: "B5", startTimeBooking: null, endTimeBooking: null },
  // ];

  // ----- Thêm xe -----
  const handleAddCar = (car) => {
    if (!selectedSlot) return false;

    // Booking gốc từ server
    const baseBookings = parkingSpots
      .filter((p) => p.name === selectedSlot)
      .filter((p) => p.startTimeBooking && p.endTimeBooking)
      .map((p) => ({
        id: p.id,
        startTimeBooking: p.startTimeBooking,
        endTimeBooking: p.endTimeBooking,
      }));

    // Booking user đang thêm
    const uiBookings = booked[selectedSlot] || [];

    const existing = [...baseBookings, ...uiBookings];

    const s2 = new Date(car.startTimeBooking).getTime();
    const e2 = new Date(car.endTimeBooking).getTime();

    // Kiểm tra trùng
    const isOverlap = existing.some((b) => {
      const s1 = new Date(b.startTimeBooking).getTime();
      const e1 = new Date(b.endTimeBooking).getTime();
      return s1 <= e2 && s2 <= e1;
    });

    if (isOverlap) return false; // báo cho BookingForm

    // Thêm booking mới
    const spot = parkingSpots.find((p) => p.name === selectedSlot);
    const newCar = {
      ...car,
      spotId: spot.id,
      id: `${spot.id}-${Date.now()}`, // unique
      spotName: selectedSlot,
    };

    setCars((prev) => [...prev, newCar]);
    setBooked((prev) => ({
      ...prev,
      [selectedSlot]: [...(prev[selectedSlot] || []), newCar],
    }));

    return true;
  };

  // ----- Xoá xe -----
  const handleRemoveCar = (index) => {
    const car = cars[index];
    setCars((prev) => prev.filter((_, i) => i !== index));
    setBooked((prev) => {
      const copy = { ...prev };
      copy[car.spotName] = copy[car.spotName].filter((c) => c.id !== car.id);
      return copy;
    });
  };

  return (
    <div className="grid grid-cols-12 gap-6 p-6">
      <div className="col-span-4">
        <h2 className="mb-2 flex items-center gap-2 text-xl font-bold">
          <Car size={22} className="text-blue-600" />
          Danh sách chỗ đậu
        </h2>
        <SlotList
          slots={slots}
          selectedSlot={selectedSlot}
          cars={cars}
          booked={booked}
          bookingMode={bookingMode}
          onSelectSlot={setSelectedSlot}
        />
      </div>

      <div className="col-span-8 flex flex-col gap-6">
        <div className="flex items-center justify-between">
          <div className="flex gap-3">
            <button
              className={`cursor-pointer rounded border px-3 py-2 ${
                bookingMode === "week"
                  ? "bg-blue-600 text-white hover:bg-blue-700"
                  : "bg-white hover:bg-gray-50"
              }`}
              onClick={() => setBookingMode("week")}
            >
              Theo tuần
            </button>
            <button
              className={`cursor-pointer rounded border px-3 py-2 ${
                bookingMode === "month"
                  ? "bg-blue-600 text-white hover:bg-blue-700"
                  : "bg-white hover:bg-gray-50"
              }`}
              onClick={() => setBookingMode("month")}
            >
              Theo tháng
            </button>
          </div>
          <CartIcon cars={cars} onRemoveCar={handleRemoveCar} />
        </div>

        {selectedSlot ? (
          <BookingForm
            slot={selectedSlot}
            bookingMode={bookingMode}
            onAddCar={handleAddCar}
            booked={booked[selectedSlot]}
          />
        ) : (
          <div className="flex h-full items-center justify-center text-gray-500">
            <div className="text-center">
              <Car size={40} className="mx-auto mb-3 text-gray-400" />
              <p className="text-lg">Chọn 1 chỗ để bắt đầu đặt xe</p>
              <p className="mt-1 text-sm text-gray-400">
                Bãi có {slots.length} vị trí đang hoạt động
              </p>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
