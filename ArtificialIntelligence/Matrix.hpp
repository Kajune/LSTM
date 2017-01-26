#pragma once

#include "Vector.hpp"

namespace LSTM {

	template <std::size_t Column, std::size_t Row, typename DataType = float>
	class Matrix {
		std::array<Vector<Row, DataType>, Column>	m_data;
	public:
		//ctor
		Matrix(const std::array<DataType, Row * Column> data) {
			for (std::size_t i = 0; i < Column; i++) {
				for (std::size_t j = 0; j < Row; j++) {
					m_data.at(i).at(j) = data.at(i * Row + j);
				}
			}
		}
		Matrix(const std::array<Vector<Row, DataType>, Column>& data) {
			for (std::size_t i = 0; i < Column; i++) {
				m_data.at(i) = data.at(i);
			}
		}
		Matrix(DataType initialData = 0) {
			fill(initialData);
		}
		Matrix(const std::initializer_list<DataType>& list) {
			operator=(list);
		}

		//copy ctor
		Matrix(const Matrix& rhs) {
			operator=(rhs);
		}
		Matrix(Matrix&& rhs) {
			operator=(std::move(rhs));
		}

		//operator=
		Matrix& operator=(const Matrix& rhs) {
			m_data = rhs.m_data;
			return *this;
		}
		Matrix& operator=(Matrix&& rhs) {
			m_data = std::move(rhs.m_data);
			return *this;
		}
		Matrix& operator=(const std::initializer_list<DataType>& list) {
			fill(0);
			auto it = list.begin();
			for (std::size_t i = 0; i < Column; i++) {
				for (std::size_t j = 0; j < Row && it != list.end(); j++) {
					m_data.at(i).at(j) = *it;
					it++;
				}
			}
			return *this;
		}

		//operator+, +=
		Matrix& operator+=(const Matrix& rhs) {
			for (std::size_t i = 0; i < Column; i++) {
				m_data.at(i) += rhs.m_data.at(i);
			}
			return *this;
		}
		Matrix operator+(const Matrix& rhs) const {
			Matrix temp(*this);
			temp += rhs;
			return temp;
		}

		//operator-, -=
		Matrix operator-() const {
			Matrix temp(*this);
			temp *= -1;
			return temp;
		}
		Matrix& operator-=(const Matrix& rhs) {
			*this += -rhs;
			return *this;
		}
		Matrix operator-(const Matrix& rhs) const {
			Matrix temp(*this);
			temp -= rhs;
			return temp;
		}

		//operator*, *=
		Matrix& operator*=(DataType rhs) {
			std::for_each(m_data.begin(), m_data.end(), [&](auto& x) {x *= rhs; });
			return *this;
		}
		Matrix operator*(DataType rhs) const {
			Matrix temp(*this);
			temp *= rhs;
			return temp;
		}
		template <std::size_t Row2>
		Matrix<Column, Row2> operator*(const Matrix<Row, Row2>& rhs) const {
			Matrix<Column, Row2> temp;
			for (std::size_t i = 0; i < Column; i++) {
				for (std::size_t j = 0; j < Row2; j++) {
					for (std::size_t k = 0; k < Row; k++) {
						temp.at(i, j) += at(i, k) * rhs.at(k, j);
					}
				}
			}
			return temp;
		}
		Vector<Column, DataType> operator*(const Vector<Row, DataType>& rhs) const {
			Vector<Column, DataType> temp;
			for (std::size_t i = 0; i < Column; i++) {
				temp.at(i) = DotProduct(at(i), rhs);
			}
			return temp;
		}

		//operator/, /=
		Matrix& operator/=(DataType rhs) {
			std::for_each(m_data.begin(), m_data.end(), [&](auto& x) {x /= rhs; });
			return *this;
		}
		Matrix operator/(DataType rhs) const {
			Matrix temp(*this);
			temp /= rhs;
			return temp;
		}

		//utility
		DataType& at(std::size_t column, std::size_t row) {
			return m_data.at(column).at(row);
		}
		DataType at(std::size_t column, std::size_t row) const {
			return m_data.at(column).at(row);
		}
		Vector<Row, DataType>& at(std::size_t column) {
			return m_data.at(column);
		}
		Vector<Row, DataType> at(std::size_t column) const {
			return m_data.at(column);
		}
		Matrix& fill(DataType value) {
			std::for_each(m_data.begin(), m_data.end(), [&](auto& x) {x.fill(value); });
			return *this;
		}
		template <typename Functor>
		Matrix transform(Functor func) const {
			Matrix temp;
			for (std::size_t i = 0; i < Column; i++) {
				temp.m_data.at(i).transform(func);
			}
			return temp;
		}
	};

	template <std::size_t Column, std::size_t Row, typename DataType>
	Matrix<Column, Row, DataType> DirectProduct
	(const Vector<Column, DataType>& lhs, const Vector<Row, DataType>& rhs) {
		Matrix<Column, Row, DataType> temp;
		for (std::size_t i = 0; i < Column; i++) {
			temp.at(i) = lhs.at(i) * rhs;
		}
		return temp;
	}

	template <std::size_t Column, std::size_t Row, typename DataType>
	Matrix<Row, Column, DataType> Transpose(const Matrix<Column, Row, DataType>& rhs) {
		Matrix<Row, Column, DataType> temp;
		for (std::size_t i = 0; i < Column; i++) {
			for (std::size_t j = 0; j < Row; j++) {
				temp.at(j, i) = rhs.at(i, j);
			}
		}
		return temp;
	}

}

