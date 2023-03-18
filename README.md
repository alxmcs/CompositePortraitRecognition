## Facial Recongnition via Composite and Photographic Portrait Comparison
НИР "**Разработка и исследование алгоритма идентификации человека по композитному портрету с использованием переноса стиля**"

### Общая постановка задачи:  
>*Решается задача распознавания человека по его натуралистическому портретному изображению, при этом в базе данных известных человек имеются только их композитные портреты.*

### Практический пример:  
>*На большинстве подъездов в городе установлены видеодомофоны. Мониторинг их видеопотока может быть полезен для поиска преступников. Часто в распоряжении правоохранительных органов отсутствует фотография разыскиваемого лица, однако может иметься фоторобот, составленный со слов потерпевших/свидетелей. Сопоставление полученных с камеры домофона портретных изображений с фотороботами может помочь в установлении местонахождения такого подозреваемого.* 
<p align="center">
<img src="https://user-images.githubusercontent.com/70561974/154902264-fd205b9e-e5e1-47b8-861e-3d985b82a391.png"/>
</p>

### Допущения:  
- Во входном видеопотоке присутствуют только фронтальные изображения человеческого лица,
- Лица людей в видеопотоке видны полностью и не закрыты масками/шарфами/солнечными очками и т.д.

### Приблизительный пайплайн:  
<p align="center">
<img src="https://user-images.githubusercontent.com/70561974/154902595-ad8ba7b4-1820-4ce8-85d7-10d36249dc89.png"/>
</p>
(Оранжевым выделен блок, над реализацией которого необходимо будет еще подумать, остальное можно реализовывать хоть сейчас, благо библиотеки для этого есть)  

### Выступления на конференциях:  
1. LXXII Молодёжная научная конференция, посвящённая 80-летию КуАИ-СГАУ-Самарского университета и 115-летию со дня рождения академика С. П. Королёва, Самара, 05.04.2022 - 07.04.2022   
2. Международная научно-техническая конференция «Перспективные информационные технологии (ПИТ-2022)», Самара, 18.04.2022 - 21.04.2022  
3. LXXIII Молодёжная научная конференция Самарского университета, Самара, 05.04.2023 - 07.04.2023
4. IX Международная конференция и молодежная школа "Информационные техноллогии и нанотехнологии (ITNT-2023)", Самара, 17.04.2023 - 21.04.2023 
  
### Публикации:  
1. Максимов А.И., Родин В.А. Исследование эффективности методов переноса стиля для задачи сопоставления натуралистичных изображений и набросков // Международная научно-техническая конференция «Перспективные информационные технологии (ПИТ-2022)». — 2022. — С. 181-184  
2. Родин В.А., Максимов А.И. Исследование влияния переноса стиля на качество сопоставления фотографического и композитного портрета // IX Международная конференция и молодежная школа "Информационные технологии и нанотехнологии (ИТНТ-2023)". — 2023.  [принято в печать]
3. Rodin V.A., Maksimov A.I. Style transfer effectiveness for forensic sketch and photo matching // 2023 9th International Conference on Information Technology and Nanotechnology, ITNT 2023. — 2023. [принято в печать]
