"use client";

import { useState, useEffect } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";

export default function UserForm({ user, onSave }) {
  const [username, setUsername] = useState(user?.username || "");
  const [email, setEmail] = useState(user?.email || "");
  const [phoneNumber, setPhoneNumber] = useState(user?.phoneNumber || "");
  const [roleName, setRoleName] = useState(user?.role?.name || "USER");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");

  useEffect(() => {
    setUsername(user?.username || "");
    setEmail(user?.email || "");
    setPhoneNumber(user?.phoneNumber || "");
    setRoleName(user?.role?.name || "USER");
    setPassword("");
    setError("");
  }, [user]);

  const handleSubmit = (e) => {
    e.preventDefault();

    if (!username || !email || (!user && !password)) {
      setError("Vui lòng điền đầy đủ thông tin bắt buộc");
      return;
    }

    const userData = {
      id: user?.id || 0,
      username,
      email,
      phoneNumber,
      role: { id: roleName === "ADMIN" ? 3 : 1, name: roleName },
      ...(user ? {} : { password }),
    };

    onSave(userData);
  };

  return (
    <form onSubmit={handleSubmit} className="space-y-4">
      <div className="flex flex-col gap-1">
        <Label>Username</Label>
        <Input value={username} onChange={(e) => setUsername(e.target.value)} required />
      </div>
      <div className="flex flex-col gap-1">
        <Label>Email</Label>
        <Input type="email" value={email} onChange={(e) => setEmail(e.target.value)} required />
      </div>
      {!user && (
        <div className="flex flex-col gap-1">
          <Label>Password</Label>
          <Input
            type="password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            required
          />
        </div>
      )}
      <div className="flex flex-col gap-1">
        <Label>Phone</Label>
        <Input value={phoneNumber} onChange={(e) => setPhoneNumber(e.target.value)} />
      </div>
      <div className="flex flex-col gap-1">
        <Label>Role</Label>
        <select
          value={roleName}
          onChange={(e) => setRoleName(e.target.value)}
          className="w-full rounded border px-2 py-1"
        >
          <option value="USER">USER</option>
          <option value="ADMIN">ADMIN</option>
        </select>
      </div>

      {error && <p className="text-red-500">{error}</p>}
      <div className="flex justify-end space-x-2">
        <Button type="submit" className="bg-green-600 text-white hover:bg-green-700">Lưu</Button>
      </div>
    </form>
  );
}
