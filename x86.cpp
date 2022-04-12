#include <iostream>
#include <fstream>
#include <windows.h>
#include <nmmintrin.h>

using namespace std;

float** generateMatrix(int n)
{
    float** m = new float* [n];
    for (int i = 0; i < n; i++)
    {
        m[i] = new float[n];
    }
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            m[i][j] = rand() % 10;
        }
    }
    return m;
}

void GE_S_N(float** m, int n)
{
    for (int k = 0; k < n; k++)
    {
        for (int j = k + 1; j < n; j++)
        {
            m[k][j] = m[k][j] / m[k][k];
        }
        m[k][k] = 1.0;
        for (int i = k + 1; i < n; i++)
        {
            for (int j = k + 1; j < n; j++)
            {
                m[i][j] -= m[i][k] * m[k][j];
            }
            m[i][k] = 0;
        }
    }
}

void GE_P_SSE(float** m, int n)
{
    __m128 va, vt, vaik, vakj, vaij, vx;
    for (int k = 0; k < n; k++)
    {
        vt = _mm_set1_ps(m[k][k]);
        int j = 0;
        for (j = k + 1; j + 4 < n; j += 4)
        {
            if (j % 4 != 0)
            {
                m[k][j] = m[k][j] / m[k][k];
                j -= 3;
                continue;
            }
            va = _mm_load_ps(&m[k][j]);
            va = _mm_div_ps(va, vt);
            _mm_store_ps(&m[k][j], va);
        }
        for (j; j < n; j++)
        {
            m[k][j] = m[k][j] / m[k][k];
        }
        m[k][k] = 1.0;
        for (int i = k + 1; i < n; i++)
        {
            vaik = _mm_set1_ps(m[i][k]);
            for (j = k + 1; j + 4 < n; j += 4)
            {
                if (j % 4 != 0)
                {
                    m[i][j] -= m[i][k] * m[k][j];
                    j -= 3;
                    continue;
                }
                vakj = _mm_load_ps(&m[k][j]);
                vaij = _mm_load_ps(&m[i][j]);
                vx = _mm_mul_ps(vakj, vaik);
                vaij = _mm_sub_ps(vaij, vx);
                _mm_store_ps(&m[i][j], vaij);
            }
            for (j; j < n; j++)
            {
                m[i][j] -= m[i][k] * m[k][j];
            }
            m[i][k] = 0;
        }
    }
}


int main()
{
    ofstream out("output.txt");
    for (int n = 200; n <= 4000; n += 400)
    {
        float** m = generateMatrix(n);
        out << n << "\t";

        long long head1, tail1, freq1;
        QueryPerformanceFrequency((LARGE_INTEGER*)&freq1);
        QueryPerformanceCounter((LARGE_INTEGER*)&head1);
        GE_S_N(m, n);
        QueryPerformanceCounter((LARGE_INTEGER*)&tail1);
        out << (tail1 - head1) * 1000.0 / freq1 << "\t";

        long long head2, tail2, freq2;
        QueryPerformanceFrequency((LARGE_INTEGER*)&freq2);
        QueryPerformanceCounter((LARGE_INTEGER*)&head2);
        GE_P_SSE(m, n);
        QueryPerformanceCounter((LARGE_INTEGER*)&tail2);
        out << (tail2 - head2) * 1000.0 / freq2;
        out << endl;

        for (int i = 0; i < n; i++)
        {
            delete[] m[i];
        }
        delete[] m;
    }
    out.close();
    return 0;
}
