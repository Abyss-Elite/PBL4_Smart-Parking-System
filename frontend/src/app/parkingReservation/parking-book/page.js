"use client";
import { useState } from "react";
import ParkingSearchBar from "@/components/parking_reservation/parking-book/ParkingSearchBar";
import ParkingCard from "@/components/parking_reservation/parking-book/ParkingCardParkingCard";

export default function ParkingPage() {
  const [results, setResults] = useState([]);

  const sampleData = [
    { id: 1, name: "Bãi đậu xe A - Trung tâm", price: 30000 },
    { id: 2, name: "Bãi đậu xe B - Gần sân bay", price: 45000 },
    { id: 3, name: "Bãi đậu xe C - Ven biển", price: 50000 },
  ];

  const handleSearch = (quantity) => {
    // API search → bạn thay bằng fetch
    setResults(sampleData);
  };

  return (
    <div className="mx-auto mt-8 max-w-4xl px-4">
      <ParkingSearchBar onSearch={handleSearch} />

      <div className="mt-6 space-y-4">
        {results.map((item) => (
          <ParkingCard
            key={item.id}
            name={item.name}
            price={item.price}
            onSelect={() => alert("Bạn chọn: " + item.name)}
          />
        ))}
      </div>
    </div>
  );
}
