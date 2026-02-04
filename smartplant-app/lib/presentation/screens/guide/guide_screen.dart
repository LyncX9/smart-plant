import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';
import '../../../core/constants/colors.dart';

class GuideScreen extends StatelessWidget {
  const GuideScreen({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: AppColors.background,
      appBar: AppBar(
        title: const Text('User Guide'),
        backgroundColor: Colors.white,
        elevation: 0,
      ),
      body: ListView(
        padding: const EdgeInsets.all(20),
        children: [
          _buildSection(
            context,
            title: 'Tentang Aplikasi',
            icon: Icons.info_outline,
            color: Colors.blue,
            content: 'SmartPlant Vision adalah aplikasi berbasis AI untuk mendeteksi kesehatan tanaman padi. Aplikasi ini menganalisis pola urat daun (vein morphometry) dan bintik penyakit untuk memberikan skor kesehatan yang akurat.',
          ),
          const SizedBox(height: 20),
          _buildSection(
            context,
            title: 'Cara Kerja',
            icon: Icons.memory,
            color: Colors.purple,
            content: '1. Mengambil gambar daun (3-7 citra).\n'
                '2. AI memproses setiap citra untuk mendeteksi urat daun dan lesi penyakit.\n'
                '3. Algoritma menggabungkan hasil dari semua gambar.\n'
                '4. Memberikan skor kesehatan (0-100) dan klasifikasi kondisi.',
          ),
          const SizedBox(height: 20),
          _buildSection(
            context,
            title: 'Tanaman yang Didukung',
            icon: Icons.local_florist,
            color: Colors.green,
            content: 'Saat ini SmartPlant Vision HANYA dioptimalkan untuk:\n\n'
                '✅ Padi (Rice Leaf)\n\n'
                'Fitur untuk tanaman lain (jangung, tomat, dll) sedang dalam pengembangan. '
                'Jika Anda memindai daun selain padi, akurasi hasil tidak terjamin atau akan muncul peringatan "Tidak Terdeteksi".',
          ),
          const SizedBox(height: 20),
          _buildSection(
            context,
            title: 'Cara Menggunakan',
            icon: Icons.touch_app,
            color: AppColors.primaryGreen,
            content: '1. Buka menu "Analyze Plant".\n'
                '2. Tekan tombol kamera untuk ambil foto atau pilih dari galeri.\n'
                '3. Pastikan foto daun jelas, tidak buram, dan pencahayaan cukup.\n'
                '4. Kumpulkan minimal 3 foto daun berbeda.\n'
                '5. Tekan "Analyze" dan tunggu hasil.',
          ),
          const SizedBox(height: 20),
          _buildSection(
            context,
            title: 'Cara Membaca Output',
            icon: Icons.analytics,
            color: Colors.orange,
            content: '• Health Score: Indikator kesehatan (0-100). Makin tinggi makin sehat.\n'
                '• Condition: Healthy (Sehat), Diseased (Sakit), atau Uncertain.\n'
                '• Vein Analysis: Visualisasi struktur urat daun (untuk edukasi).\n'
                '• Lesion Count: Jumlah bintik penyakit yang terdeteksi.',
          ),
          const SizedBox(height: 32),
          Center(
            child: Text(
              'v1.0.0 (Production Build)',
              style: GoogleFonts.outfit(
                color: AppColors.textSecondary,
                fontSize: 12,
              ),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildSection(
    BuildContext context, {
    required String title,
    required String content,
    required IconData icon,
    required Color color,
  }) {
    return Container(
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(16),
        boxShadow: [
          BoxShadow(
            color: Colors.black.withOpacity(0.05),
            blurRadius: 10,
            offset: const Offset(0, 4),
          ),
        ],
      ),
      child: Theme(
        data: Theme.of(context).copyWith(dividerColor: Colors.transparent),
        child: ExpansionTile(
          initiallyExpanded: true,
          leading: Container(
            padding: const EdgeInsets.all(8),
            decoration: BoxDecoration(
              color: color.withOpacity(0.1),
              borderRadius: BorderRadius.circular(8),
            ),
            child: Icon(icon, color: color),
          ),
          title: Text(
            title,
            style: GoogleFonts.outfit(
              fontWeight: FontWeight.bold,
              fontSize: 16,
              color: AppColors.textPrimary,
            ),
          ),
          children: [
            Padding(
              padding: const EdgeInsets.fromLTRB(20, 0, 20, 20),
              child: Text(
                content,
                style: GoogleFonts.outfit(
                  fontSize: 14,
                  height: 1.5,
                  color: AppColors.textSecondary,
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }
}
