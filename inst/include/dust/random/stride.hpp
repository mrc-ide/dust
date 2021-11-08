//

// // We already basically have this elsewhere
// template <typename T, typename U>
// class strided_container {
// public:
//   strided_container(T x, size_t stride) : x_(x), stride_(stride) {}
//   U& operator[](size_t i) {
//     x_[i * stride];
//   }
// private:
//   T x_;
//   size_t stride_;
// }
