#pragma once

#include <array>
#include <algorithm>

namespace LSTM {

	template <std::size_t Size, typename DataType = float>
	class Vector {
		std::array<DataType, Size>	m_data;
	public:
		//ctor
		Vector(const std::array<DataType, Size>& data) : m_data(data) {
		}
		Vector(DataType initialData = 0) {
			m_data.fill(initialData);
		}
		Vector(const std::initializer_list<DataType>& list) {
			operator=(list);
		}

		//copy ctor
		Vector(const Vector& rhs) {
			operator=(rhs);
		}
		Vector(Vector&& rhs) {
			operator=(std::move(rhs));
		}

		//operator=
		Vector& operator=(const Vector& rhs) {
			m_data = rhs.m_data;
			return *this;
		}
		Vector& operator=(Vector&& rhs) {
			m_data = std::move(rhs.m_data);
			return *this;
		}
		Vector& operator=(const std::initializer_list<DataType>& list) {
			fill(0);
			auto it = list.begin();
			for (std::size_t i = 0; i < Size && it != list.end(); i++) {
				m_data.at(i) = *it;
				it++;
			}
			return *this;
		}

		//operator+, +=
		Vector& operator+=(const Vector& rhs) {
			for (std::size_t i = 0; i < Size; i++) {
				m_data.at(i) += rhs.m_data.at(i);
			}
			return *this;
		}
		Vector operator+(const Vector& rhs) const {
			Vector temp(*this);
			temp += rhs;
			return temp;
		}

		//operator-, -=
		Vector operator-() const {
			Vector temp(*this);
			temp *= -1;
			return temp;
		}
		Vector& operator-=(const Vector& rhs) {
			*this += -rhs;
			return *this;
		}
		Vector operator-(const Vector& rhs) const {
			Vector temp(*this);
			temp -= rhs;
			return temp;
		}

		//operator*, *=
		Vector& operator*=(DataType rhs) {
			std::for_each(m_data.begin(), m_data.end(), [&](DataType& x) { x *= rhs; });
			return *this;
		}
		Vector operator*(DataType rhs) const {
			Vector temp(*this);
			temp *= rhs;
			return temp;
		}
		friend Vector operator*(DataType lhs, Vector rhs) {
			return rhs * lhs;
		}

		//operator/, /=
		Vector& operator/=(DataType rhs) {
			std::for_each(m_data.begin(), m_data.end(), [&](DataType& x) { x /= rhs; });
			return *this;
		}
		Vector operator/(DataType rhs) const {
			Vector temp(*this);
			temp /= rhs;
			return temp;
		}

		//utility
		DataType& at(std::size_t i) {
			return m_data.at(i);
		}
		DataType at(std::size_t i) const {
			return m_data.at(i);
		}
		Vector& fill(DataType value, std::size_t index = 0) {
			for (std::size_t i = index; i < Size; i++) {
				m_data.at(i) = value;
			}
			return *this;
		}
		template <typename Functor>
		Vector transform(Functor func) const {
			Vector temp;
			for (std::size_t i = 0; i < Size; i++) {
				temp.m_data.at(i) = func(m_data.at(i));
			}
			return temp;
		}
	};

	//dot product
	template <std::size_t Size, typename DataType>
	DataType DotProduct(const Vector<Size, DataType>& lhs, const Vector<Size, DataType>& rhs) {
		DataType sum{ 0 };
		for (std::size_t i = 0; i < Size; i++) {
			sum += lhs.at(i) * rhs.at(i);
		}
		return sum;
	}

	//hadamard product
	template <std::size_t Size, typename DataType>
	Vector<Size, DataType> HadamardProduct(const Vector<Size, DataType>& lhs, const Vector<Size, DataType>& rhs) {
		Vector<Size, DataType> temp;
		for (std::size_t i = 0; i < Size; i++) {
			temp.at(i) = lhs.at(i) * rhs.at(i);
		}
		return temp;
	}

	//hadamard quotient
	template <std::size_t Size, typename DataType>
	Vector<Size, DataType> HadamardQuotient(const Vector<Size, DataType>& lhs, const Vector<Size, DataType>& rhs) {
		Vector<Size, DataType> temp;
		for (std::size_t i = 0; i < Size; i++) {
			temp.at(i) = lhs.at(i) / rhs.at(i);
		}
		return temp;
	}

}
