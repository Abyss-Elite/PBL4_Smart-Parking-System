"use client";
import SlotCard from "./SlotCard";

export default function SlotList({ slots, selectedSlot, cars, booked, bookingMode, onSelectSlot }) {
  const yourCarsForSlot = (slot) => cars.filter((c) => c.spotName === slot);

  return (
    <div className="space-y-4">
      <div className="grid max-h-[85vh] grid-cols-2 gap-3 overflow-y-auto pr-2">
        {slots.map((s) => (
          <SlotCard
            key={s}
            slot={s}
            booked={booked[s] || []}
            bookingMode={bookingMode}
            userCars={yourCarsForSlot(s)}
            onSelectCar={onSelectSlot}
            selected={selectedSlot}
          />
        ))}
      </div>
    </div>
  );
}
