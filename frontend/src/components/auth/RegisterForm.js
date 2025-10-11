"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import { Alert, AlertDescription } from "@/components/ui/alert";
import authAPI from "@/api/auth/authApi";
import PATH from "@/routes/PATH";

export default function RegisterForm() {
  const router = useRouter();
  const [username, setUsername] = useState("");
  const [email, setEmail] = useState("");
  const [password1, setPassword1] = useState("");
  const [password2, setPassword2] = useState("");
  const [errorMessage, setErrorMessage] = useState("");

  const handleRegister = async () => {
    if (password1 !== password2) {
      setErrorMessage("Mật khẩu xác nhận không khớp!");
      return;
    }

    try {
      const res = await authAPI.register({ username, email, password, role });

      if (res.status === 200 || res.status === 201) {
        setErrorMessage("");
        router.push(PATH.LOGIN);
      } else {
        setErrorMessage(res.data.message || "Đăng ký thất bại!");
      }
    } catch (err) {
      setErrorMessage(err.response?.data?.message || "Không thể đăng ký. Vui lòng thử lại!");
    }
  };

  return (
    <Card className="animate-fadeIn w-full max-w-md rounded-2xl bg-white p-6 shadow-xl">
      <CardHeader>
        <CardTitle className="text-center text-2xl font-bold text-gray-800">Đăng ký</CardTitle>
      </CardHeader>
      <CardContent className="space-y-3">
        <div>
          <label className="text-sm font-medium">Tên đăng nhập</label>
          <Input
            placeholder="Tên đăng nhập"
            value={username}
            onChange={(e) => setUsername(e.target.value)}
          />
        </div>

        <div>
          <label className="text-sm font-medium">Email</label>
          <Input
            placeholder="Email"
            type="email"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
          />
        </div>

        <div>
          <label className="text-sm font-medium">Mật khẩu</label>
          <Input
            placeholder="Mật khẩu"
            type="password"
            value={password1}
            onChange={(e) => setPassword1(e.target.value)}
          />
        </div>

        <div>
          <label className="text-sm font-medium">Xác nhận mật khẩu</label>
          <Input
            placeholder="Xác nhận mật khẩu"
            type="password"
            value={password2}
            onChange={(e) => setPassword2(e.target.value)}
          />
        </div>

        {errorMessage && (
          <Alert variant="destructive">
            <AlertDescription>{errorMessage}</AlertDescription>
          </Alert>
        )}

        <Button onClick={handleRegister} className="w-full bg-green-500 hover:bg-green-600">
          Đăng ký
        </Button>

        <div className="mt-3 text-center">
          <a href="/login" className="text-blue-600 transition-colors hover:text-green-500">
            Bạn đã có tài khoản? Đăng nhập
          </a>
        </div>
      </CardContent>
    </Card>
  );
}
