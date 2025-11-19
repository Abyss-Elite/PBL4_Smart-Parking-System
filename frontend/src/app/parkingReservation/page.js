"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import BookingForm from "@/components/parking_reservation/parking-book/BookingForm";
import SlotList from "@/components/parking_reservation/parking-book/SlotList";
import CartIcon from "@/components/parking_reservation/parking-book/CartIcon";
import { Car } from "lucide-react";

export default function SlotPage() {
  const router = useRouter();
  const [selectedSlot, setSelectedSlot] = useState(null);
  const [cars, setCars] = useState([]);

  const slots = Array.from({ length: 50 }, (_, i) => `A${i + 1}`);
  const booked = {
    A1: [
      { start: "08:00", end: "09:30" },
      { start: "14:00", end: "16:00" },
    ],
    A3: [{ start: "10:00", end: "11:00" }],
  };

  const handleAddCar = (car) => setCars((prev) => [...prev, car]);
  const handleRemoveCar = (index) => setCars((prev) => prev.filter((_, i) => i !== index));
  
  const handleSubmit = () => {
    if (cars.length === 0) return alert("Bạn chưa chọn xe nào!");
    localStorage.setItem("parking_booking", JSON.stringify(cars));
    router.push("/parkingReservation/confirm");
  };

  const handleUpdateCar = (updatedCars) => {
    setCars(updatedCars);
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
          onSelectSlot={setSelectedSlot}
        />
      </div>

      <div className="col-span-8 flex flex-col gap-6">
        <div className="flex justify-end">
          
          <CartIcon
            cars={cars}
            onRemoveCar={handleRemoveCar}
            onUpdateCar={handleUpdateCar}
            onSubmit={handleSubmit}
          />
        </div>

        {selectedSlot ? (
          <BookingForm
            slot={selectedSlot}
            bookedTimes={[
              ...(booked[selectedSlot] || []),
              ...cars
                .filter((c) => c.slot === selectedSlot)
                .map((c) => ({ start: c.start, end: c.end })),
            ]}
            selectedCars={cars.filter((c) => c.slot === selectedSlot)}
            onAddCar={handleAddCar}
            onClickSubmit={handleSubmit}
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
