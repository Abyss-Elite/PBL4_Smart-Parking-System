package com.example.backend.model;

import jakarta.persistence.*;
import jakarta.validation.constraints.Email;
import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.Pattern;
import jakarta.validation.constraints.Size;
import com.fasterxml.jackson.annotation.JsonIgnore;
import java.util.List;

@Entity
@Table(name = "users")
public class User {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;
  
    @Column(unique = true, nullable = false)
    @NotBlank(message = "Tên đăng nhập không được để trống")
    private String username;

    @Column(unique = true, nullable = false)
    @NotBlank(message = "Số điện thoại không được để trống")
    private String phoneNumber;

    @Column(nullable = false)
    @NotBlank(message = "Mật khẩu không được để trống")
    @Size(min = 8, message = "Mật khẩu phải ít nhất 8 ký tự")
    @Pattern(
        regexp = ".*[!@#$%^&*(),.?\":{}|<>].*",
        message = "Mật khẩu phải ít nhất 1 ký tự đặc biệt"
    )
    private String password;

    @NotBlank(message = "Email không được để trống")
    @Email(message = "Email không hợp lệ")
    @Column(nullable = false, unique = true)
    private String email;

    @Column(nullable = true)
    private String avaUrl;

    @Column(nullable = false)
    private String status;

    @Column(nullable = false)
    private boolean isDelete;

    @ManyToOne
    @JoinColumn(name = "role_id")
    private Role role;

    @OneToMany(mappedBy = "owner", cascade = CascadeType.ALL, orphanRemoval = true)
    @JsonIgnore
    private List<Car> cars;

    // ======== Getter & Setter thủ công ========

    public Long getId() {
        return id;
    }

    public void setId(Long id) {
        this.id = id;
    }

    public String getUsername() {
        return username;
    }

    public void setUsername(String username) {
        this.username = username;
    }

    public String getPhoneNumber() {
        return phoneNumber;
    }

    public void setPhoneNumber(String phoneNumber) {
        this.phoneNumber = phoneNumber;
    }

    public String getPassword() {
        return password;
    }

    public void setPassword(String password) {
        this.password = password;
    }

    public String getEmail() {
        return email;
    }

    public void setEmail(String email) {
        this.email = email;
    }

    public String getAvaUrl() {
        return avaUrl;
    }

     public String getStatus() { return status; }
    public void setStatus(String status) { this.status = status; }

    public boolean isDelete() { return isDelete; }
    public void setDelete(boolean delete) { isDelete = delete; }

    public void setAvaUrl(String avaUrl) {
        this.avaUrl = avaUrl;
    }

    public List<Car> getCars() {
        return cars;
    }

    public void setCars(List<Car> cars) {
        this.cars = cars;
    }

    public Role getRole() { return role; }
    public void setRole(Role roles) { this.role = roles; }
}
