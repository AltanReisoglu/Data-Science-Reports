"""Gerçek masaüstü testinde ortaya çıkan üç hatanın regresyon kilidi.

Üçü de birim testlerinden kaçmıştı; ancak gerçek bir X11 oturumunda koşarken
görüldü. Hepsi "koruma çalışıyor sanılırken sessizce kapalıydı" tipinde.
"""

import unittest

from cua_lab.safety import Blocked, SafetyPolicy


class SayacMuhasebesi(unittest.TestCase):
    """`charge()` yalnız GERÇEKTEN gönderilen girdiyi saymalı.

    Eskiden `_gonder()` başında çağrılıyordu: engellenen `Delete` bile sert
    tavandan düşüyor ve `rapor()` hiçbir şey gönderilmediği hâlde
    "5 gerçek girdi" diyordu. Güvenlik raporunun yanlış sayı vermesi,
    korumanın kendisi kadar ciddi.
    """

    def test_engellenen_eylem_sayaci_artirmaz(self):
        p = SafetyPolicy(allow_input=True)
        for tus in ("Delete", "shift+Delete", "alt+F4"):
            # Sandbox'ın gerçek akışı: check → Blocked → note_blocked.
            # `charge()` ARTIK check'ten SONRA, yani hiç çalışmıyor.
            try:
                p.check_key(tus); p.charge()
            except Blocked as e:
                p.note_blocked(e.kural, e.detay)
            else:
                self.fail(f"{tus} engellenmedi")
        self.assertIn("0 gercek girdi", p.rapor())
        self.assertIn("3 eylem ENGELLENDI", p.rapor())

    def test_gecen_eylem_sayilir(self):
        p = SafetyPolicy(allow_input=True)
        p.check_key("Return"); p.charge()
        self.assertIn("1 gercek girdi", p.rapor())


class ModernTerminaller(unittest.TestCase):
    """`terminal|konsole|xterm|...` deseni modern terminalleri kaçırıyordu.

    GNOME Console ve Ptyxis, Ubuntu/Fedora'da varsayılan terminal oldu ve
    adlarında "terminal" geçmiyor. Sınıf kontrolü bozukken (bkz. aşağıdaki
    sınıf) bu iki kat tehlikeliydi.
    """

    def _engellenmeli(self, sinif):
        with self.assertRaises(Blocked, msg=f"{sinif} engellenmedi"):
            SafetyPolicy(allow_input=True).check_window("x", sinif)

    def test_gnome_console(self):   self._engellenmeli("org.gnome.Console")
    def test_ptyxis(self):          self._engellenmeli("ptyxis Ptyxis")
    def test_ghostty(self):         self._engellenmeli("com.mitchellh.ghostty")
    def test_wezterm(self):         self._engellenmeli("org.wezfurlong.wezterm")
    def test_terminator(self):      self._engellenmeli("terminator Terminator")
    def test_gnome_terminal(self):  self._engellenmeli("gnome-terminal-server")

    def test_tarayici_devtools_engellenmez(self):
        """Genel `console` deseni kullanılmamalı: tarayıcının
        "Console - DevTools" başlığını yakalayıp tarayıcı görevlerini
        boşuna keserdi."""
        SafetyPolicy(allow_input=True).check_window("Console - DevTools", "brave Brave")


class SinifOkuma(unittest.TestCase):
    """`xdotool getwindowclassname` bu makinede HİÇ YOK ("Unknown command").

    `aktif_sinif()` bunu yutup `""` dönüyordu — yani WM_CLASS kontrolü
    sessizce kapalıydı ve koruma yalnız başlığa düşmüştü. `xprop` yedeği
    eklendi; burada `xprop` çıktısının ayrıştırılması kilitleniyor.
    """

    def test_wm_class_ayristirma(self):
        import re
        cikti = 'WM_CLASS(STRING) = "gnome-terminal-server", "Gnome-terminal"'
        sinif = " ".join(re.findall(r'"([^"]*)"', cikti))
        self.assertEqual(sinif, "gnome-terminal-server Gnome-terminal")
        with self.assertRaises(Blocked):
            SafetyPolicy(allow_input=True).check_window("altan@laptop", sinif)


if __name__ == "__main__":
    unittest.main()
