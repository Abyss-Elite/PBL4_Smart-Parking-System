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

export function ParkingSearchFilter({ setSearchLicensePlate, onclick }) {
  return (
    <div className="mb-4 flex flex-wrap items-center gap-3">
      <Input
        placeholder="Nhập biển số..."
        className="w-48"
        onChange={(e) => setSearchLicensePlate(e.target.value)}
      />

      <Select>
        <SelectTrigger className="w-32">
          <SelectValue placeholder="Trạng thái" />
        </SelectTrigger>
        <SelectContent>
          <SelectItem value="parking">Đang đỗ</SelectItem>
          <SelectItem value="left">Đã rời</SelectItem>
        </SelectContent>
      </Select>

      <Button variant="default" onClick={onclick} className="cursor-pointer">
        Tìm kiếm
      </Button>
      <Button variant="outline" className="cursor-pointer">
        Làm mới
      </Button>
    </div>
  );
}
