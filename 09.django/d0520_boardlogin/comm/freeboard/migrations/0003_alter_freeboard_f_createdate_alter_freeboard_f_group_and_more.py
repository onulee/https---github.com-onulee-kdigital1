# Generated by Django 4.0.4 on 2022-05-20 06:50

import datetime
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('freeboard', '0002_alter_freeboard_f_createdate_and_more'),
    ]

    operations = [
        migrations.AlterField(
            model_name='freeboard',
            name='f_createdate',
            field=models.DateTimeField(blank=True, default=datetime.datetime(2022, 5, 20, 15, 50, 55, 488756)),
        ),
        migrations.AlterField(
            model_name='freeboard',
            name='f_group',
            field=models.IntegerField(default=0),
        ),
        migrations.AlterField(
            model_name='freeboard',
            name='f_indent',
            field=models.IntegerField(default=0),
        ),
        migrations.AlterField(
            model_name='freeboard',
            name='f_step',
            field=models.IntegerField(default=0),
        ),
        migrations.AlterField(
            model_name='freeboard',
            name='f_updatedate',
            field=models.DateTimeField(blank=True, default=datetime.datetime(2022, 5, 20, 15, 50, 55, 488756)),
        ),
    ]
