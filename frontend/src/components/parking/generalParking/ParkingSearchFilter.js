"use client";

import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";

export function ParkingSearchFilter({
  searchLicensePlate,
  setSearchLicensePlate,
  onReset,
  onSearch,
}) {
  return (
    <div className="mb-4 flex flex-wrap items-center gap-3">
      <Input
        placeholder="Nhập biển số..."
        className="w-48"
        value={searchLicensePlate}
        onChange={(e) => setSearchLicensePlate(e.target.value)}
      />

      <Button variant="default" onClick={onSearch} className="cursor-pointer">
        Tìm kiếm
      </Button>
      <Button
        variant="outline"
        className="cursor-pointer"
        onClick={() => {
          setSearchLicensePlate("");
          onReset();
        }}
      >
        Làm mới
      </Button>
    </div>
  );
}
